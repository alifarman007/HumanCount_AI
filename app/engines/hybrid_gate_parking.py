"""
Hybrid Gate + Parking Counting Engine (Module 7)
Line crossing + parking zone detection for factory scenarios.
"""

import math
import time
from typing import Dict, List, Set, Optional

from app.engines.base import CountingEngineBase
from app.model.data_structures import (
    TrackState,
    CountEvent,
    CountDirection,
    ROIConfiguration,
)
from app.model.track_manager import TrackManager
from app.geometry.polygon_math import PolygonZone


class HybridGateParkingEngine(CountingEngineBase, engine_type="hybrid_gate_parking"):
    """
    Hybrid counting for factory gates with parking areas.
    
    Combines:
    - Tripwire at gate for walking entry/exit
    - Parking zone for vehicle passenger spawn/vanish detection
    
    Counting logic:
    - Walk IN through gate → Entry (Walk)
    - Appear in parking zone (spawn) → Entry (Vehicle)
    - Walk OUT through gate → Exit (Walk)
    - Vanish in parking zone (disappeared for N seconds) → Exit (Vehicle)
    """
    
    ENGINE_NAME = "Hybrid Gate + Parking"
    ENGINE_DESCRIPTION = "Line crossing + parking zone detection. Best for factory gates with vehicle parking."
    
    def __init__(self, config: ROIConfiguration):
        # Gate line (tripwire)
        self._gate_line: Optional[tuple] = None  # (p1, p2)
        self._gate_entry_side = "side_a"
        self._gate_line_length = 0.0
        
        # Parking zone
        self._parking_zone: PolygonZone = None
        
        # Vanish detection settings
        self._vanish_timeout_frames = 90  # Default 3 seconds at 30fps
        
        # Track states for gate crossing
        self._track_gate_side: Dict[int, str] = {}
        
        # Track states for parking zone
        self._tracks_in_parking: Dict[int, bool] = {}
        self._track_last_seen_in_parking: Dict[int, int] = {}  # track_id -> frame number
        self._spawned_tracks: Set[int] = set()  # Tracks that spawned in parking
        
        # Frame counter
        self._frame_count = 0
        
        super().__init__(config)
    
    def _validate_config(self):
        """Validate hybrid configuration."""
        pass
    
    def _setup_geometry(self):
        """Setup gate and parking zone from config."""
        # Get vanish timeout from config
        if self._config.module_specific:
            timeout_sec = self._config.module_specific.get('vanish_timeout_seconds', 3.0)
            fps = self._config.module_specific.get('fps', 30)
            self._vanish_timeout_frames = int(timeout_sec * fps)
        
        if self._config.lines:
            line = self._config.lines[0]
            self._gate_entry_side = getattr(line, 'entry_direction', 'side_a')
    
    def set_gate_line_pixels(self, p1: tuple, p2: tuple, entry_side: str = "side_a"):
        """Set gate line in pixel coordinates."""
        self._gate_line = (p1, p2)
        self._gate_entry_side = entry_side
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        self._gate_line_length = math.sqrt(dx * dx + dy * dy)
    
    def set_parking_zone_pixels(self, vertices: List[tuple]):
        """Set parking zone in pixel coordinates."""
        if len(vertices) >= 3:
            self._parking_zone = PolygonZone(vertices)
    
    def set_vanish_timeout(self, seconds: float, fps: int = 30):
        """Set vanish detection timeout."""
        self._vanish_timeout_frames = int(seconds * fps)
    
    def _cross_product(self, point: tuple) -> float:
        """Calculate cross product for gate line."""
        if not self._gate_line:
            return 0
        
        x, y = point
        x1, y1 = self._gate_line[0]
        x2, y2 = self._gate_line[1]
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    
    def _get_gate_side(self, point: tuple, hysteresis: float) -> str:
        """Get which side of gate line a point is on."""
        if not self._gate_line or self._gate_line_length == 0:
            return "center"
        
        cross = self._cross_product(point)
        distance = abs(cross) / self._gate_line_length
        
        if distance < hysteresis:
            return "center"
        
        return "side_a" if cross > 0 else "side_b"
    
    def process_tracks(
        self,
        tracks: Dict[int, TrackState],
        track_manager: TrackManager
    ) -> List[CountEvent]:
        """Process tracks for hybrid counting."""
        events = []
        self._frame_count += 1
        hysteresis = self._debounce.hysteresis_distance
        
        current_track_ids = set(tracks.keys())
        
        for track_id, track in tracks.items():
            if not track.current_position:
                continue
            
            point = track.current_position
            is_new_track = track_manager.is_new_track(track_id)
            in_cooldown = track_manager.is_in_cooldown(track_id)
            
            # === PARKING ZONE LOGIC ===
            if self._parking_zone:
                is_in_parking = self._parking_zone.contains(point)
                was_in_parking = self._tracks_in_parking.get(track_id, False)
                
                # Spawn detection: new track appears inside parking zone
                if is_new_track and is_in_parking:
                    if track_id not in self._spawned_tracks and not track.counted_spawn:
                        # Person spawned in parking (got out of car)
                        event = self._create_count_event(
                            track_id=track_id,
                            direction=CountDirection.SPAWN,
                            position=point,
                            metadata={'type': 'vehicle', 'location': 'parking'}
                        )
                        events.append(event)
                        self._emit_count_event(event)
                        self._counts['entry_vehicle'] = self._counts.get('entry_vehicle', 0) + 1
                        
                        track_manager.set_counted_spawn(track_id, True)
                        self._spawned_tracks.add(track_id)
                
                # Update parking tracking
                if is_in_parking:
                    self._track_last_seen_in_parking[track_id] = self._frame_count
                
                self._tracks_in_parking[track_id] = is_in_parking
            
            # === GATE LINE LOGIC ===
            if self._gate_line and not in_cooldown:
                current_side = self._get_gate_side(point, hysteresis)
                
                if current_side != "center":
                    prev_side = self._track_gate_side.get(track_id)
                    
                    if prev_side is None:
                        self._track_gate_side[track_id] = current_side
                    elif prev_side != current_side:
                        # Gate crossing detected
                        if prev_side == self._gate_entry_side:
                            # Entry through gate (walking in)
                            if not track.counted_entry:
                                event = self._create_count_event(
                                    track_id=track_id,
                                    direction=CountDirection.ENTRY,
                                    position=point,
                                    metadata={'type': 'walk', 'location': 'gate'}
                                )
                                events.append(event)
                                self._emit_count_event(event)
                                self._counts['entry_walk'] = self._counts.get('entry_walk', 0) + 1
                                
                                track_manager.set_counted_entry(track_id, True)
                                track_manager.set_cooldown(track_id, self._debounce.cooldown_frames)
                        else:
                            # Exit through gate (walking out)
                            if not track.counted_exit:
                                event = self._create_count_event(
                                    track_id=track_id,
                                    direction=CountDirection.EXIT,
                                    position=point,
                                    metadata={'type': 'walk', 'location': 'gate'}
                                )
                                events.append(event)
                                self._emit_count_event(event)
                                self._counts['exit_walk'] = self._counts.get('exit_walk', 0) + 1
                                
                                track_manager.set_counted_exit(track_id, True)
                                track_manager.set_cooldown(track_id, self._debounce.cooldown_frames)
                        
                        self._track_gate_side[track_id] = current_side
        
        # === VANISH DETECTION ===
        # Check for tracks that were in parking and disappeared
        if self._parking_zone:
            for track_id in list(self._track_last_seen_in_parking.keys()):
                if track_id not in current_track_ids:
                    # Track is gone
                    last_seen_frame = self._track_last_seen_in_parking[track_id]
                    was_in_parking = self._tracks_in_parking.get(track_id, False)
                    
                    frames_gone = self._frame_count - last_seen_frame
                    
                    if was_in_parking and frames_gone >= self._vanish_timeout_frames:
                        # Track vanished while in parking (got into car)
                        if track_id not in self._spawned_tracks:
                            # Only count vanish if they didn't spawn here
                            event = self._create_count_event(
                                track_id=track_id,
                                direction=CountDirection.VANISH,
                                position=(0, 0),  # Position unknown
                                metadata={'type': 'vehicle', 'location': 'parking'}
                            )
                            events.append(event)
                            self._emit_count_event(event)
                            self._counts['exit_vehicle'] = self._counts.get('exit_vehicle', 0) + 1
                        
                        # Cleanup
                        del self._track_last_seen_in_parking[track_id]
                        if track_id in self._tracks_in_parking:
                            del self._tracks_in_parking[track_id]
                        if track_id in self._spawned_tracks:
                            self._spawned_tracks.discard(track_id)
        
        # Cleanup old gate tracking
        stale_gate_ids = [tid for tid in self._track_gate_side if tid not in current_track_ids]
        for tid in stale_gate_ids:
            del self._track_gate_side[tid]
        
        return events
    
    @property
    def total_entry(self) -> int:
        """Get total entry count (walk + vehicle)."""
        return (self._counts.get('entry', 0) + 
                self._counts.get('entry_walk', 0) + 
                self._counts.get('entry_vehicle', 0))
    
    @property
    def total_exit(self) -> int:
        """Get total exit count (walk + vehicle)."""
        return (self._counts.get('exit', 0) + 
                self._counts.get('exit_walk', 0) + 
                self._counts.get('exit_vehicle', 0))
    
    def reset_counts(self):
        """Reset all counts and tracking state."""
        super().reset_counts()
        self._track_gate_side.clear()
        self._tracks_in_parking.clear()
        self._track_last_seen_in_parking.clear()
        self._spawned_tracks.clear()
        self._frame_count = 0
    
    def get_visualization_data(self) -> Dict:
        """Get hybrid data for visualization."""
        data = super().get_visualization_data()
        data['gate_line'] = self._gate_line
        data['gate_entry_side'] = self._gate_entry_side
        if self._parking_zone:
            data['parking_zone_vertices'] = self._parking_zone.vertices
        data['entry_walk'] = self._counts.get('entry_walk', 0)
        data['entry_vehicle'] = self._counts.get('entry_vehicle', 0)
        data['exit_walk'] = self._counts.get('exit_walk', 0)
        data['exit_vehicle'] = self._counts.get('exit_vehicle', 0)
        return data