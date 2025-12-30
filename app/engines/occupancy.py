"""
Real-time Occupancy Counting Engine (Module 6)
Counts people currently inside a zone (not crossings).
"""

from typing import Dict, List, Set
from enum import Enum, auto

from app.engines.base import CountingEngineBase
from app.model.data_structures import (
    TrackState,
    CountEvent,
    CountDirection,
    ROIConfiguration,
    AlertLevel,
)
from app.model.track_manager import TrackManager
from app.geometry.polygon_math import PolygonZone
from app.utils.event_bus import get_event_bus


class OccupancyEngine(CountingEngineBase, engine_type="occupancy"):
    """
    Real-time occupancy counting.
    
    Counts how many people are currently inside a zone (snapshot count).
    Supports capacity limits and alert levels.
    """
    
    ENGINE_NAME = "Real-time Occupancy"
    ENGINE_DESCRIPTION = "Count people currently inside a zone. Best for rooms, elevators, and capacity monitoring."
    
    def __init__(self, config: ROIConfiguration):
        self._zone: PolygonZone = None
        
        # Capacity settings
        self._max_capacity = 50
        self._warning_threshold = 0.8  # 80% of max
        self._critical_threshold = 0.96  # 96% of max
        
        # Current state
        self._current_occupancy = 0
        self._current_alert_level = AlertLevel.NORMAL
        self._tracks_inside: Set[int] = set()
        
        super().__init__(config)
    
    def _validate_config(self):
        """Validate occupancy configuration."""
        pass
    
    def _setup_geometry(self):
        """Setup zone from config."""
        if self._config.polygons and len(self._config.polygons) > 0:
            pass  # Will be set via set_zone_pixels
        
        # Get capacity settings from module_specific
        if self._config.module_specific:
            self._max_capacity = self._config.module_specific.get('max_capacity', 50)
            self._warning_threshold = self._config.module_specific.get('warning_threshold', 0.8)
            self._critical_threshold = self._config.module_specific.get('critical_threshold', 0.96)
    
    def set_zone_pixels(self, vertices: List[tuple]):
        """
        Set the occupancy zone in pixel coordinates.
        
        Args:
            vertices: List of (x, y) tuples defining the polygon
        """
        if len(vertices) >= 3:
            self._zone = PolygonZone(vertices)
    
    def set_capacity(self, max_capacity: int, warning_pct: float = 0.8, critical_pct: float = 0.96):
        """
        Set capacity limits.
        
        Args:
            max_capacity: Maximum allowed occupancy
            warning_pct: Percentage for warning level (0-1)
            critical_pct: Percentage for critical level (0-1)
        """
        self._max_capacity = max_capacity
        self._warning_threshold = warning_pct
        self._critical_threshold = critical_pct
    
    def _calculate_alert_level(self, occupancy: int) -> AlertLevel:
        """Calculate alert level based on occupancy."""
        if self._max_capacity <= 0:
            return AlertLevel.NORMAL
        
        ratio = occupancy / self._max_capacity
        
        if ratio > 1.0:
            return AlertLevel.EXCEEDED
        elif ratio >= self._critical_threshold:
            return AlertLevel.CRITICAL
        elif ratio >= self._warning_threshold:
            return AlertLevel.WARNING
        else:
            return AlertLevel.NORMAL
    
    def process_tracks(
        self,
        tracks: Dict[int, TrackState],
        track_manager: TrackManager
    ) -> List[CountEvent]:
        """Process tracks for occupancy counting."""
        events = []
        
        if self._zone is None:
            return events
        
        # Count tracks inside zone
        new_tracks_inside: Set[int] = set()
        
        for track_id, track in tracks.items():
            if not track.current_position:
                continue
            
            point = track.current_position
            
            if self._zone.contains(point):
                new_tracks_inside.add(track_id)
        
        # Update occupancy
        previous_occupancy = self._current_occupancy
        self._current_occupancy = len(new_tracks_inside)
        self._tracks_inside = new_tracks_inside
        
        # Update counts for display
        self._counts['current_occupancy'] = self._current_occupancy
        
        # Check for alert level change
        new_alert_level = self._calculate_alert_level(self._current_occupancy)
        
        if new_alert_level != self._current_alert_level:
            self._current_alert_level = new_alert_level
            
            # Emit alert event
            event_bus = get_event_bus()
            event_bus.emit_alert(
                alert_type=new_alert_level.name,
                message=f"Occupancy: {self._current_occupancy}/{self._max_capacity}",
                data={
                    'occupancy': self._current_occupancy,
                    'max_capacity': self._max_capacity,
                    'alert_level': new_alert_level.name
                }
            )
        
        # Emit counts updated
        self._event_bus.emit_counts_updated(self._counts)
        
        return events
    
    @property
    def current_occupancy(self) -> int:
        """Get current occupancy count."""
        return self._current_occupancy
    
    @property
    def max_capacity(self) -> int:
        """Get maximum capacity."""
        return self._max_capacity
    
    @property
    def alert_level(self) -> AlertLevel:
        """Get current alert level."""
        return self._current_alert_level
    
    @property
    def occupancy_percentage(self) -> float:
        """Get occupancy as percentage of capacity."""
        if self._max_capacity <= 0:
            return 0.0
        return (self._current_occupancy / self._max_capacity) * 100
    
    def reset_counts(self):
        """Reset occupancy tracking."""
        super().reset_counts()
        self._current_occupancy = 0
        self._current_alert_level = AlertLevel.NORMAL
        self._tracks_inside.clear()
    
    def get_visualization_data(self) -> Dict:
        """Get occupancy data for visualization."""
        data = super().get_visualization_data()
        if self._zone:
            data['zone_vertices'] = self._zone.vertices
        data['current_occupancy'] = self._current_occupancy
        data['max_capacity'] = self._max_capacity
        data['alert_level'] = self._current_alert_level.name
        data['occupancy_percentage'] = self.occupancy_percentage
        return data