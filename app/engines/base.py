"""
Base Counting Engine - Abstract base class for all counting modules.
Provides auto-registration, common interface, and shared utilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Type, ClassVar, Optional
import time

from app.model.data_structures import (
    TrackState,
    CountEvent,
    CountDirection,
    ROIConfiguration,
    DebounceConfig
)
from app.model.track_manager import TrackManager
from app.utils.event_bus import get_event_bus


class CountingEngineBase(ABC):
    """
    Abstract base class for all counting engines.
    
    Features:
    - Auto-registration via __init_subclass__
    - Common counting interface
    - Built-in event bus integration
    - Debouncing support
    
    Subclasses must implement:
    - _validate_config(): Validate module-specific configuration
    - _setup_geometry(): Setup geometric primitives
    - process_tracks(): Main counting logic
    """
    
    # Class-level registry of all engine types
    _registry: ClassVar[Dict[str, Type['CountingEngineBase']]] = {}
    
    # Engine type identifier (set by subclass)
    ENGINE_TYPE: ClassVar[str] = "base"
    ENGINE_NAME: ClassVar[str] = "Base Engine"
    ENGINE_DESCRIPTION: ClassVar[str] = "Base counting engine"
    
    def __init_subclass__(cls, engine_type: str = None, **kwargs):
        """Auto-register subclasses."""
        super().__init_subclass__(**kwargs)
        if engine_type:
            cls._registry[engine_type] = cls
            cls.ENGINE_TYPE = engine_type
    
    @classmethod
    def get_engine_class(cls, engine_type: str) -> Type['CountingEngineBase']:
        """Get engine class by type string."""
        if engine_type not in cls._registry:
            raise ValueError(f"Unknown engine type: {engine_type}. Available: {list(cls._registry.keys())}")
        return cls._registry[engine_type]
    
    @classmethod
    def create_engine(cls, engine_type: str, config: ROIConfiguration) -> 'CountingEngineBase':
        """Factory method to create engine instance."""
        engine_class = cls.get_engine_class(engine_type)
        return engine_class(config)
    
    @classmethod
    def list_engines(cls) -> List[Dict[str, str]]:
        """List all registered engine types with metadata."""
        engines = []
        for engine_type, engine_class in cls._registry.items():
            engines.append({
                'type': engine_type,
                'name': getattr(engine_class, 'ENGINE_NAME', engine_type),
                'description': getattr(engine_class, 'ENGINE_DESCRIPTION', '')
            })
        return engines
    
    def __init__(self, config: ROIConfiguration):
        """
        Initialize counting engine.
        
        Args:
            config: ROI configuration for this engine
        """
        self._config = config
        self._debounce = config.debounce
        self._event_bus = get_event_bus()
        
        # Counting state
        self._counts = {
            'entry': 0,
            'exit': 0,
            'entry_walk': 0,
            'exit_walk': 0,
            'entry_vehicle': 0,
            'exit_vehicle': 0,
            'violations': 0,
            'current_occupancy': 0
        }
        
        # Validate and setup
        self._validate_config()
        self._setup_geometry()
        
        # Emit ready status
        self._event_bus.emit_engine_status(self.ENGINE_TYPE, "initialized")
    
    @abstractmethod
    def _validate_config(self):
        """
        Validate engine-specific configuration.
        Raise ValueError if configuration is invalid.
        """
        pass
    
    @abstractmethod
    def _setup_geometry(self):
        """
        Setup geometric primitives (lines, polygons, etc.) from config.
        Called after validation.
        """
        pass
    
    @abstractmethod
    def process_tracks(
        self,
        tracks: Dict[int, TrackState],
        track_manager: TrackManager
    ) -> List[CountEvent]:
        """
        Process all tracks and return count events.
        
        This is called once per frame with all current track states.
        
        Args:
            tracks: Dictionary of track_id -> TrackState
            track_manager: TrackManager instance for updating flags
        
        Returns:
            List of CountEvent objects for events detected this frame
        """
        pass
    
    @property
    def counts(self) -> Dict[str, int]:
        """Get current count totals."""
        return self._counts.copy()
    
    @property
    def total_entry(self) -> int:
        """Get total entry count."""
        return self._counts.get('entry', 0) + self._counts.get('entry_walk', 0) + self._counts.get('entry_vehicle', 0)
    
    @property
    def total_exit(self) -> int:
        """Get total exit count."""
        return self._counts.get('exit', 0) + self._counts.get('exit_walk', 0) + self._counts.get('exit_vehicle', 0)
    
    @property
    def current_inside(self) -> int:
        """Get current count inside (entry - exit)."""
        return max(0, self.total_entry - self.total_exit)
    
    def reset_counts(self):
        """Reset all counters to zero."""
        for key in self._counts:
            self._counts[key] = 0
        self._event_bus.emit_counts_updated(self._counts)
    
    def _emit_count_event(self, event: CountEvent):
        """Emit count event and update internal counts."""
        # Update counts based on direction
        if event.direction == CountDirection.ENTRY:
            self._counts['entry'] += 1
        elif event.direction == CountDirection.EXIT:
            self._counts['exit'] += 1
        elif event.direction == CountDirection.SPAWN:
            self._counts['entry_vehicle'] += 1
        elif event.direction == CountDirection.VANISH:
            self._counts['exit_vehicle'] += 1
        
        # Emit events
        self._event_bus.emit_count(event)
        self._event_bus.emit_counts_updated(self._counts)
    
    def _create_count_event(
        self,
        track_id: int,
        direction: CountDirection,
        position: tuple,
        metadata: Dict = None
    ) -> CountEvent:
        """Helper to create a CountEvent."""
        return CountEvent(
            track_id=track_id,
            direction=direction,
            engine_type=self.ENGINE_TYPE,
            timestamp=time.time(),
            position=position,
            metadata=metadata or {}
        )
    
    def get_visualization_data(self) -> Dict:
        """
        Get data for UI visualization.
        Override in subclasses for engine-specific visualization.
        
        Returns:
            Dict with visualization data (lines, polygons, counts, etc.)
        """
        return {
            'engine_type': self.ENGINE_TYPE,
            'counts': self._counts,
            'config': self._config.to_dict() if self._config else {}
        }
    
    def cleanup(self):
        """Cleanup resources. Override if needed."""
        self._event_bus.emit_engine_status(self.ENGINE_TYPE, "stopped")