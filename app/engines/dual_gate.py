"""
Dual-Gate Logic Counting Engine (Module 3)
Two separate lines with directional enforcement for one-way security.
"""

import math
from typing import Dict, List, Optional

from app.engines.base import CountingEngineBase
from app.model.data_structures import (
    TrackState,
    CountEvent,
    CountDirection,
    ROIConfiguration,
)
from app.model.track_manager import TrackManager


class DualGateEngine(CountingEngineBase, engine_type="dual_gate"):
    """
    Dual-gate counting with directional enforcement.
    
    Two separate lines:
    - Entry gate: Only counts valid entries (can flag wrong-way violations)
    - Exit gate: Only counts valid exits (can flag wrong-way violations)
    """
    
    ENGINE_NAME = "Dual-Gate Logic"
    ENGINE_DESCRIPTION = "Two separate lines with directional enforcement. Best for one-way security doors."
    
    def __init__(self, config: ROIConfiguration):
        # Gate lines: (p1, p2)
        self._entry_gate: Optional[tuple] = None
        self._exit_gate: Optional[tuple] = None
        
        # Entry direction for each gate (which side people should come FROM)
        self._entry_gate_direction = "side_a"
        self._exit_gate_direction = "side_a"
        
        # Track violations
        self._violations = 0
        self._enable_violations = True
        
        # Track last side per gate
        self._track_entry_gate_side: Dict[int, str] = {}
        self._track_exit_gate_side: Dict[int, str] = {}
        
        super().__init__(config)
    
    def _validate_config(self):
        """Validate dual-gate configuration."""
        # Will be configured via set methods
        pass
    
    def _setup_geometry(self):
        """Setup gates from config."""
        if self._config.lines and len(self._config.lines) >= 2:
            entry_line = self._config.lines[0]
            exit_line = self._config.lines[1]
            self._entry_gate_direction = getattr(entry_line, 'entry_direction', 'side_a')
            self._exit_gate_direction = getattr(exit_line, 'entry_direction', 'side_a')
    
    def set_entry_gate_pixels(self, p1: tuple, p2: tuple, entry_from: str = "side_a"):
        """Set entry gate line in pixel coordinates."""
        self._entry_gate = (p1, p2)
        self._entry_gate_direction = entry_from
    
    def set_exit_gate_pixels(self, p1: tuple, p2: tuple, exit_from: str = "side_a"):
        """Set exit gate line in pixel coordinates."""
        self._exit_gate = (p1, p2)
        self._exit_gate_direction = exit_from
    
    def set_enable_violations(self, enable: bool):
        """Enable or disable violation detection."""
        self._enable_violations = enable
    
    def _cross_product(self, line_p1: tuple, line_p2: tuple, point: tuple) -> float:
        """Calculate cross product."""
        x, y = point
        x1, y1 = line_p1
        x2, y2 = line_p2
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    
    def _get_line_length(self, p1: tuple, p2: tuple) -> float:
        """Get line length."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def _get_side(self, gate: tuple, point: tuple, hysteresis: float) -> str:
        """Get which side of a gate a point is on."""
        p1, p2 = gate
        length = self._get_line_length(p1, p2)
        
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
        """Process tracks for dual-gate crossing."""
        events = []
        hysteresis = self._debounce.hysteresis_distance
        
        for track_id, track in tracks.items():
            if track_manager.is_in_cooldown(track_id):
                continue
            
            if not track.current_position:
                continue
            
            point = track.current_position
            
            # Check entry gate
            if self._entry_gate and not track.counted_entry_gate:
                current_side = self._get_side(self._entry_gate, point, hysteresis)
                
                if current_side != "center":
                    prev_side = self._track_entry_gate_side.get(track_id)
                    
                    if prev_side is None:
                        self._track_entry_gate_side[track_id] = current_side
                    elif prev_side != current_side:
                        # Crossing detected
                        if prev_side == self._entry_gate_direction:
                            # Valid entry (coming from correct side)
                            event = self._create_count_event(
                                track_id=track_id,
                                direction=CountDirection.ENTRY,
                                position=point,
                                metadata={'gate': 'entry', 'valid': True}
                            )
                            events.append(event)
                            self._emit_count_event(event)
                            track_manager.set_counted_entry_gate(track_id, True)
                        else:
                            # Wrong way through entry gate!
                            if self._enable_violations:
                                self._violations += 1
                                event = self._create_count_event(
                                    track_id=track_id,
                                    direction=CountDirection.EXIT,  # Leaving through entry
                                    position=point,
                                    metadata={'gate': 'entry', 'valid': False, 'violation': True}
                                )
                                events.append(event)
                                self._emit_count_event(event)
                        
                        track_manager.set_cooldown(track_id, self._debounce.cooldown_frames)
                        self._track_entry_gate_side[track_id] = current_side
            
            # Check exit gate
            if self._exit_gate and not track.counted_exit_gate:
                current_side = self._get_side(self._exit_gate, point, hysteresis)
                
                if current_side != "center":
                    prev_side = self._track_exit_gate_side.get(track_id)
                    
                    if prev_side is None:
                        self._track_exit_gate_side[track_id] = current_side
                    elif prev_side != current_side:
                        # Crossing detected
                        if prev_side == self._exit_gate_direction:
                            # Valid exit
                            event = self._create_count_event(
                                track_id=track_id,
                                direction=CountDirection.EXIT,
                                position=point,
                                metadata={'gate': 'exit', 'valid': True}
                            )
                            events.append(event)
                            self._emit_count_event(event)
                            track_manager.set_counted_exit_gate(track_id, True)
                        else:
                            # Wrong way through exit gate!
                            if self._enable_violations:
                                self._violations += 1
                                event = self._create_count_event(
                                    track_id=track_id,
                                    direction=CountDirection.ENTRY,  # Entering through exit
                                    position=point,
                                    metadata={'gate': 'exit', 'valid': False, 'violation': True}
                                )
                                events.append(event)
                                self._emit_count_event(event)
                        
                        track_manager.set_cooldown(track_id, self._debounce.cooldown_frames)
                        self._track_exit_gate_side[track_id] = current_side
        
        return events
    
    @property
    def violations(self) -> int:
        """Get violation count."""
        return self._violations
    
    def reset_counts(self):
        """Reset counts including violations."""
        super().reset_counts()
        self._violations = 0
        self._track_entry_gate_side.clear()
        self._track_exit_gate_side.clear()
    
    def get_visualization_data(self) -> Dict:
        """Get gate data for visualization."""
        data = super().get_visualization_data()
        data['entry_gate'] = self._entry_gate
        data['exit_gate'] = self._exit_gate
        data['violations'] = self._violations
        return data