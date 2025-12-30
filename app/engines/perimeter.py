"""
Virtual Perimeter Counting Engine (Module 5)
Single polygon boundary detection with spawn detection.
"""

from typing import Dict, List, Set

from app.engines.base import CountingEngineBase
from app.model.data_structures import (
    TrackState,
    CountEvent,
    CountDirection,
    ROIConfiguration,
)
from app.model.track_manager import TrackManager
from app.geometry.polygon_math import PolygonZone


class PerimeterEngine(CountingEngineBase, engine_type="perimeter"):
    """
    Virtual perimeter counting using a single polygon.
    
    Counts:
    - Entry: Person moves from outside to inside
    - Exit: Person moves from inside to outside
    - Spawn: Person first appears inside (e.g., car passenger)
    """
    
    ENGINE_NAME = "Virtual Perimeter"
    ENGINE_DESCRIPTION = "Single zone boundary detection. Best for kiosks, counters, and display areas."
    
    def __init__(self, config: ROIConfiguration):
        self._zone: PolygonZone = None
        self._track_inside: Dict[int, bool] = {}  # Track ID -> is inside
        self._enable_spawn_detection = True
        
        super().__init__(config)
    
    def _validate_config(self):
        """Validate perimeter configuration."""
        # Will be configured via set_zone_pixels()
        pass
    
    def _setup_geometry(self):
        """Setup zone from config."""
        if self._config.polygons and len(self._config.polygons) > 0:
            vertices = self._config.polygons[0].vertices
            if vertices:
                # Convert normalized to absolute if needed (will be set properly via set_zone_pixels)
                pass
    
    def set_zone_pixels(self, vertices: List[tuple]):
        """
        Set the perimeter zone in pixel coordinates.
        
        Args:
            vertices: List of (x, y) tuples defining the polygon
        """
        if len(vertices) >= 3:
            self._zone = PolygonZone(vertices)
    
    def set_enable_spawn_detection(self, enable: bool):
        """Enable or disable spawn detection."""
        self._enable_spawn_detection = enable
    
    def process_tracks(
        self,
        tracks: Dict[int, TrackState],
        track_manager: TrackManager
    ) -> List[CountEvent]:
        """Process tracks for perimeter crossing."""
        events = []
        
        if self._zone is None:
            return events
        
        min_frames = self._debounce.min_frames_in_zone
        
        for track_id, track in tracks.items():
            if track_manager.is_in_cooldown(track_id):
                continue
            
            if not track.current_position:
                continue
            
            point = track.current_position
            is_inside = self._zone.contains(point)
            
            # Check if this is a new track
            is_new = track_id not in self._track_inside
            
            if is_new:
                self._track_inside[track_id] = is_inside
                
                # Spawn detection: first appearance inside the zone
                if is_inside and self._enable_spawn_detection and not track.counted_spawn:
                    event = self._create_count_event(
                        track_id=track_id,
                        direction=CountDirection.SPAWN,
                        position=point,
                        metadata={'spawn': True}
                    )
                    events.append(event)
                    self._emit_count_event(event)
                    track_manager.set_counted_spawn(track_id, True)
                    track_manager.set_cooldown(track_id, self._debounce.cooldown_frames)
                
                continue
            
            was_inside = self._track_inside[track_id]
            
            # Check for transition
            if was_inside != is_inside:
                if is_inside and not track.counted_entry:
                    # Entered the zone
                    direction = CountDirection.ENTRY
                    track_manager.set_counted_entry(track_id, True)
                elif not is_inside and not track.counted_exit:
                    # Exited the zone
                    direction = CountDirection.EXIT
                    track_manager.set_counted_exit(track_id, True)
                else:
                    self._track_inside[track_id] = is_inside
                    continue
                
                event = self._create_count_event(
                    track_id=track_id,
                    direction=direction,
                    position=point
                )
                events.append(event)
                self._emit_count_event(event)
                track_manager.set_cooldown(track_id, self._debounce.cooldown_frames)
            
            self._track_inside[track_id] = is_inside
        
        # Cleanup old tracks
        active_ids = set(tracks.keys())
        stale_ids = [tid for tid in self._track_inside if tid not in active_ids]
        for tid in stale_ids:
            del self._track_inside[tid]
        
        return events
    
    def get_current_inside_count(self) -> int:
        """Get number of tracks currently inside the zone."""
        return sum(1 for is_in in self._track_inside.values() if is_in)
    
    def reset_counts(self):
        """Reset counts and tracking state."""
        super().reset_counts()
        self._track_inside.clear()
    
    def get_visualization_data(self) -> Dict:
        """Get zone data for visualization."""
        data = super().get_visualization_data()
        if self._zone:
            data['zone_vertices'] = self._zone.vertices
        data['current_inside'] = self.get_current_inside_count()
        return data