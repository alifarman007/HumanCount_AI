"""
Zone Transition Counting Engine (Module 4)
Two adjacent zones requiring sequential passage (state machine).
"""

from typing import Dict, List, Optional
from enum import Enum, auto

from app.engines.base import CountingEngineBase
from app.model.data_structures import (
    TrackState,
    CountEvent,
    CountDirection,
    ROIConfiguration,
    ZoneState,
)
from app.model.track_manager import TrackManager
from app.geometry.polygon_math import PolygonZone


class ZoneTransitionEngine(CountingEngineBase, engine_type="zone_transition"):
    """
    Zone transition counting with state machine.
    
    Two adjacent zones (A and B):
    - Valid ENTRY: OUTSIDE → ZONE_A → ZONE_B
    - Valid EXIT: ZONE_B → ZONE_A → OUTSIDE
    
    Requires sequential passage through both zones to count.
    Prevents false counts from people just passing by.
    """
    
    ENGINE_NAME = "Zone Transition"
    ENGINE_DESCRIPTION = "Two adjacent zones requiring sequential passage. Best for wide lobbies and malls."
    
    def __init__(self, config: ROIConfiguration):
        self._zone_a: PolygonZone = None  # Detection zone
        self._zone_b: PolygonZone = None  # Confirmation zone
        
        # Track states: track_id -> ZoneState
        self._track_zone_states: Dict[int, ZoneState] = {}
        # Frames in current state for debouncing
        self._track_state_frames: Dict[int, int] = {}
        
        super().__init__(config)
    
    def _validate_config(self):
        """Validate zone transition configuration."""
        pass
    
    def _setup_geometry(self):
        """Setup zones from config."""
        if self._config.polygons and len(self._config.polygons) >= 2:
            pass  # Will be set via set_zones_pixels
    
    def set_zones_pixels(self, zone_a_vertices: List[tuple], zone_b_vertices: List[tuple]):
        """
        Set both zones in pixel coordinates.
        
        Args:
            zone_a_vertices: Detection zone vertices
            zone_b_vertices: Confirmation zone vertices
        """
        if len(zone_a_vertices) >= 3:
            self._zone_a = PolygonZone(zone_a_vertices)
        if len(zone_b_vertices) >= 3:
            self._zone_b = PolygonZone(zone_b_vertices)
    
    def _get_current_zone(self, point: tuple) -> ZoneState:
        """Determine which zone a point is in."""
        if self._zone_b and self._zone_b.contains(point):
            return ZoneState.ZONE_B
        elif self._zone_a and self._zone_a.contains(point):
            return ZoneState.ZONE_A
        else:
            return ZoneState.OUTSIDE
    
    def _is_valid_entry_transition(self, prev_state: ZoneState, curr_state: ZoneState) -> bool:
        """Check if this is a valid entry transition."""
        # Valid entry sequence: OUTSIDE -> ZONE_A -> ZONE_B
        if prev_state == ZoneState.ZONE_A and curr_state == ZoneState.ZONE_B:
            return True
        return False
    
    def _is_valid_exit_transition(self, prev_state: ZoneState, curr_state: ZoneState) -> bool:
        """Check if this is a valid exit transition."""
        # Valid exit sequence: ZONE_B -> ZONE_A -> OUTSIDE
        if prev_state == ZoneState.ZONE_A and curr_state == ZoneState.OUTSIDE:
            return True
        return False
    
    def process_tracks(
        self,
        tracks: Dict[int, TrackState],
        track_manager: TrackManager
    ) -> List[CountEvent]:
        """Process tracks for zone transitions."""
        events = []
        
        if self._zone_a is None or self._zone_b is None:
            return events
        
        min_frames = self._debounce.min_frames_in_zone
        
        for track_id, track in tracks.items():
            if track_manager.is_in_cooldown(track_id):
                continue
            
            if not track.current_position:
                continue
            
            point = track.current_position
            current_zone = self._get_current_zone(point)
            
            # Initialize new tracks
            if track_id not in self._track_zone_states:
                self._track_zone_states[track_id] = current_zone
                self._track_state_frames[track_id] = 0
                
                # If starting in ZONE_B, they're already inside
                if current_zone == ZoneState.ZONE_B:
                    track_manager.set_zone_state(track_id, ZoneState.ZONE_B)
                continue
            
            prev_zone = self._track_zone_states[track_id]
            
            # Check if zone changed
            if prev_zone == current_zone:
                # Same zone, increment frame counter
                self._track_state_frames[track_id] = self._track_state_frames.get(track_id, 0) + 1
                continue
            
            # Zone changed - check if stable enough (debounce)
            frames_in_prev = self._track_state_frames.get(track_id, 0)
            
            if frames_in_prev < min_frames:
                # Not stable enough, could be noise
                self._track_state_frames[track_id] = 0
                self._track_zone_states[track_id] = current_zone
                continue
            
            # Valid transition, check for entry/exit
            if self._is_valid_entry_transition(prev_zone, current_zone):
                if not track.counted_entry:
                    event = self._create_count_event(
                        track_id=track_id,
                        direction=CountDirection.ENTRY,
                        position=point,
                        metadata={'from_zone': prev_zone.name, 'to_zone': current_zone.name}
                    )
                    events.append(event)
                    self._emit_count_event(event)
                    track_manager.set_counted_entry(track_id, True)
                    track_manager.set_cooldown(track_id, self._debounce.cooldown_frames)
            
            elif self._is_valid_exit_transition(prev_zone, current_zone):
                # Check if track was in ZONE_B before ZONE_A
                if track.zone_state == ZoneState.ZONE_B or prev_zone == ZoneState.ZONE_A:
                    if not track.counted_exit:
                        event = self._create_count_event(
                            track_id=track_id,
                            direction=CountDirection.EXIT,
                            position=point,
                            metadata={'from_zone': prev_zone.name, 'to_zone': current_zone.name}
                        )
                        events.append(event)
                        self._emit_count_event(event)
                        track_manager.set_counted_exit(track_id, True)
                        track_manager.set_cooldown(track_id, self._debounce.cooldown_frames)
            
            # Update stored state
            self._track_zone_states[track_id] = current_zone
            self._track_state_frames[track_id] = 0
            track_manager.set_zone_state(track_id, current_zone)
        
        # Cleanup stale tracks
        active_ids = set(tracks.keys())
        stale_ids = [tid for tid in self._track_zone_states if tid not in active_ids]
        for tid in stale_ids:
            del self._track_zone_states[tid]
            if tid in self._track_state_frames:
                del self._track_state_frames[tid]
        
        return events
    
    def reset_counts(self):
        """Reset counts and state tracking."""
        super().reset_counts()
        self._track_zone_states.clear()
        self._track_state_frames.clear()
    
    def get_visualization_data(self) -> Dict:
        """Get zone data for visualization."""
        data = super().get_visualization_data()
        if self._zone_a:
            data['zone_a_vertices'] = self._zone_a.vertices
        if self._zone_b:
            data['zone_b_vertices'] = self._zone_b.vertices
        return data