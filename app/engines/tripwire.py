"""
Tripwire Counting Engine (Module 1)
Simple line crossing detection using vector cross product.
"""

from typing import Dict, List, Optional

from app.engines.base import CountingEngineBase
from app.model.data_structures import (
    TrackState,
    CountEvent,
    CountDirection,
    ROIConfiguration,
    LineConfig
)
from app.model.track_manager import TrackManager
from app.geometry.line_math import (
    get_line_side_discrete,
    detect_line_crossing,
    get_crossing_direction_as_count
)


class TripwireEngine(
    CountingEngineBase,
    engine_type="tripwire"
):
    """
    Simple tripwire counting using a single line.
    
    Counts people crossing a line in either direction.
    Uses cross product for side detection with hysteresis debouncing.
    """
    
    ENGINE_NAME = "Simple Tripwire"
    ENGINE_DESCRIPTION = "Count crossings over a single line. Best for doors, corridors, turnstiles."
    
    def __init__(self, config: ROIConfiguration):
        # Line geometry (set after parent init calls _setup_geometry)
        self._line_p1: tuple = None
        self._line_p2: tuple = None
        self._entry_direction: str = "left_to_right"
        
        super().__init__(config)
    
    def _validate_config(self):
        """Validate that we have exactly one line configured."""
        if not self._config.lines:
            raise ValueError("Tripwire engine requires at least one line")
        
        line = self._config.lines[0]
        if not line.p1 or not line.p2:
            raise ValueError("Line must have both p1 and p2 points")
    
    def _setup_geometry(self):
        """Setup line geometry from config."""
        line = self._config.lines[0]
        self._line_p1 = tuple(line.p1)
        self._line_p2 = tuple(line.p2)
        self._entry_direction = line.entry_direction
    
    def set_line_pixels(self, p1: tuple, p2: tuple):
        """
        Set line coordinates in pixel space.
        Called by UI after drawing.
        """
        self._line_p1 = p1
        self._line_p2 = p2
    
    def process_tracks(
        self,
        tracks: Dict[int, TrackState],
        track_manager: TrackManager
    ) -> List[CountEvent]:
        """
        Process all tracks for line crossing.
        
        Args:
            tracks: Current track states
            track_manager: For updating flags
        
        Returns:
            List of count events detected this frame
        """
        events = []
        hysteresis = self._debounce.hysteresis_distance
        
        for track_id, track in tracks.items():
            # Skip if in cooldown
            if track_manager.is_in_cooldown(track_id):
                continue
            
            # Need at least 2 positions for crossing detection
            if len(track.position_history) < 2:
                # Initialize line side for new tracks
                if track.current_position and track.last_line_side is None:
                    side = get_line_side_discrete(
                        self._line_p1, 
                        self._line_p2, 
                        track.current_position,
                        hysteresis
                    )
                    track_manager.set_line_side(track_id, side)
                continue
            
            curr_pos = track.current_position
            prev_pos = track.previous_position
            
            if not curr_pos or not prev_pos:
                continue
            
            # Get current side with hysteresis
            curr_side = get_line_side_discrete(
                self._line_p1,
                self._line_p2,
                curr_pos,
                hysteresis
            )
            
            # Skip if in hysteresis zone
            if curr_side == 0:
                continue
            
            # Get previous side (from stored state for stability)
            prev_side = track.last_line_side
            
            # Initialize if needed
            if prev_side is None:
                track_manager.set_line_side(track_id, curr_side)
                continue
            
            # Check for crossing
            if prev_side != 0 and prev_side != curr_side:
                # Crossing detected!
                crossing = detect_line_crossing(
                    self._line_p1,
                    self._line_p2,
                    prev_pos,
                    curr_pos,
                    hysteresis
                )
                
                if crossing:
                    # Determine if entry or exit
                    count_type = get_crossing_direction_as_count(
                        crossing,
                        self._entry_direction
                    )
                    
                    # Check if already counted
                    if count_type == "entry" and not track.counted_entry:
                        direction = CountDirection.ENTRY
                        track_manager.set_counted_entry(track_id, True)
                    elif count_type == "exit" and not track.counted_exit:
                        direction = CountDirection.EXIT
                        track_manager.set_counted_exit(track_id, True)
                    else:
                        # Already counted this direction
                        track_manager.set_line_side(track_id, curr_side)
                        continue
                    
                    # Create event
                    event = self._create_count_event(
                        track_id=track_id,
                        direction=direction,
                        position=curr_pos,
                        metadata={
                            'crossing': crossing,
                            'line_id': self._config.lines[0].id
                        }
                    )
                    
                    events.append(event)
                    self._emit_count_event(event)
                    
                    # Set cooldown
                    track_manager.set_cooldown(
                        track_id, 
                        self._debounce.cooldown_frames
                    )
            
            # Update stored side
            track_manager.set_line_side(track_id, curr_side)
        
        return events
    
    def get_visualization_data(self) -> Dict:
        """Get line data for UI overlay."""
        data = super().get_visualization_data()
        data['line'] = {
            'p1': self._line_p1,
            'p2': self._line_p2,
            'entry_direction': self._entry_direction
        }
        return data