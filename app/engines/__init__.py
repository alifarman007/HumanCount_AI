"""
Counting Engines Package

All engines auto-register via __init_subclass__.
Import them here to ensure registration.
"""

from app.engines.base import CountingEngineBase
from app.engines.tripwire import TripwireEngine
from app.engines.polyline import PolylineEngine
from app.engines.dual_gate import DualGateEngine
from app.engines.perimeter import PerimeterEngine
from app.engines.occupancy import OccupancyEngine
from app.engines.zone_transition import ZoneTransitionEngine
from app.engines.hybrid_gate_parking import HybridGateParkingEngine

__all__ = [
    'CountingEngineBase',
    'TripwireEngine',
    'PolylineEngine',
    'DualGateEngine',
    'PerimeterEngine',
    'OccupancyEngine',
    'ZoneTransitionEngine',
    'HybridGateParkingEngine',
]