"""
Event Bus - Central event dispatching using Qt Signals.
Singleton pattern ensures all components share the same bus.
"""

from PySide6.QtCore import QObject, Signal
from typing import Dict, Any

from app.model.data_structures import CountEvent


class EventBus(QObject):
    """
    Central event bus for all counting events.
    
    Uses Qt Signals for thread-safe event dispatching.
    Singleton pattern - use EventBus.instance() to get the shared instance.
    """
    
    # Signals
    count_event = Signal(object)        # CountEvent
    alert_event = Signal(dict)          # Alert data dict
    engine_status = Signal(str, str)    # engine_type, status message
    counts_updated = Signal(dict)       # Current counts dict
    frame_processed = Signal(int)       # Frame number
    error_occurred = Signal(str, str)   # error_type, message
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._initialized = True
    
    @classmethod
    def instance(cls) -> 'EventBus':
        """Get the singleton EventBus instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def emit_count(self, event: CountEvent):
        """Emit a count event."""
        self.count_event.emit(event)
    
    def emit_alert(self, alert_type: str, message: str, data: Dict = None):
        """Emit an alert event."""
        self.alert_event.emit({
            'type': alert_type,
            'message': message,
            'data': data or {}
        })
    
    def emit_counts_updated(self, counts: Dict[str, int]):
        """Emit updated counts."""
        self.counts_updated.emit(counts)
    
    def emit_engine_status(self, engine_type: str, status: str):
        """Emit engine status change."""
        self.engine_status.emit(engine_type, status)
    
    def emit_frame_processed(self, frame_number: int):
        """Emit frame processed notification."""
        self.frame_processed.emit(frame_number)
    
    def emit_error(self, error_type: str, message: str):
        """Emit error event."""
        self.error_occurred.emit(error_type, message)


# Convenience function
def get_event_bus() -> EventBus:
    """Get the shared EventBus instance."""
    return EventBus.instance()