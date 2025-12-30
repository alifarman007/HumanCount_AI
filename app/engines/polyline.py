"""
Multi-Segment Polyline Counting Engine (Module 2)
Count crossings over connected line segments (L-shaped, corners, etc.)
"""

import math
from typing import Dict, List, Tuple

from app.engines.base import CountingEngineBase
from app.model.data_structures import (
    TrackState,
    CountEvent,
    CountDirection,
    ROIConfiguration,
)
from app.model.track_manager import TrackManager


class PolylineEngine(CountingEngineBase, engine_type="polyline"):
    """
    Multi-segment polyline counting.
    
    User draws 3+ points creating connected line segments.
    Crossing ANY segment triggers a count.
    """
    
    ENGINE_NAME = "Multi-Segment Polyline"
    ENGINE_DESCRIPTION = "Count crossings over connected line segments. Best for L-shaped entrances and corners."
    
    def __init__(self, config: ROIConfiguration):
        self._segments: List[Tuple[tuple, tuple]] = []  # List of (p1, p2) tuples
        self._entry_side = "side_a"
        super().__init__(config)
    
    def _validate_config(self):
        """Validate polyline configuration."""
        if not self._config.lines:
            raise ValueError("Polyline engine requires line configuration")
    
    def _setup_geometry(self):
        """Setup polyline segments from config."""
        if self._config.lines:
            line = self._config.lines[0]
            self._entry_side = getattr(line, 'entry_direction', 'side_a')
            # Segments will be set via set_segments_pixels()
    
    def set_segments_pixels(self, points: List[tuple]):
        """
        Set polyline from list of points in pixel coordinates.
        Creates segments connecting consecutive points.
        
        Args:
            points: List of (x, y) tuples defining the polyline
        """
        self._segments = []
        for i in range(len(points) - 1):
            self._segments.append((points[i], points[i + 1]))
    
    def set_entry_side(self, side: str):
        """Set which side is entry."""
        self._entry_side = side
    
    def _cross_product(self, line_p1: tuple, line_p2: tuple, point: tuple) -> float:
        """Calculate cross product for a point relative to a line segment."""
        x, y = point
        x1, y1 = line_p1
        x2, y2 = line_p2
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    
    def _get_segment_length(self, p1: tuple, p2: tuple) -> float:
        """Calculate length of a segment."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def _get_side_for_segment(self, segment_idx: int, point: tuple, hysteresis: float) -> str:
        """Get which side of a segment a point is on."""
        if segment_idx >= len(self._segments):
            return "center"
        
        p1, p2 = self._segments[segment_idx]
        length = self._get_segment_length(p1, p2)
        
        if length == 0:
            return "center"
        
        cross = self._cross_product(p1, p2, point)
        distance = abs(cross) / length
        
        if distance < hysteresis:
            return "center"
        
        return "side_a" if cross > 0 else "side_b"
    
    def process_tracks(
        self,
        tracks: Dict[int, TrackState],
        track_manager: TrackManager
    ) -> List[CountEvent]:
        """Process tracks for polyline crossing."""
        events = []
        hysteresis = self._debounce.hysteresis_distance
        
        if not self._segments:
            return events
        
        for track_id, track in tracks.items():
            if track_manager.is_in_cooldown(track_id):
                continue
            
            if not track.current_position:
                continue
            
            point = track.current_position
            
            # Check each segment
            for seg_idx, (p1, p2) in enumerate(self._segments):
                current_side = self._get_side_for_segment(seg_idx, point, hysteresis)
                
                if current_side == "center":
                    continue
                
                # Get previous side for this segment
                prev_side = track.segment_sides.get(seg_idx)
                
                if prev_side is None:
                    track_manager.set_segment_side(track_id, seg_idx, 
                        1 if current_side == "side_a" else -1)
                    continue
                
                prev_side_str = "side_a" if prev_side > 0 else "side_b"
                
                # Check for crossing
                if prev_side_str != current_side:
                    # Determine direction
                    if prev_side_str == self._entry_side:
                        if not track.counted_entry:
                            direction = CountDirection.ENTRY
                            track_manager.set_counted_entry(track_id, True)
                        else:
                            track_manager.set_segment_side(track_id, seg_idx,
                                1 if current_side == "side_a" else -1)
                            continue
                    else:
                        if not track.counted_exit:
                            direction = CountDirection.EXIT
                            track_manager.set_counted_exit(track_id, True)
                        else:
                            track_manager.set_segment_side(track_id, seg_idx,
                                1 if current_side == "side_a" else -1)
                            continue
                    
                    event = self._create_count_event(
                        track_id=track_id,
                        direction=direction,
                        position=point,
                        metadata={'segment_index': seg_idx}
                    )
                    events.append(event)
                    self._emit_count_event(event)
                    
                    track_manager.set_cooldown(track_id, self._debounce.cooldown_frames)
                    break  # Only count once per frame
                
                track_manager.set_segment_side(track_id, seg_idx,
                    1 if current_side == "side_a" else -1)
        
        return events
    
    def get_visualization_data(self) -> Dict:
        """Get polyline data for visualization."""
        data = super().get_visualization_data()
        data['segments'] = self._segments
        data['entry_side'] = self._entry_side
        return data