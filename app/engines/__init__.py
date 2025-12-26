"""
Counting Engines Package

All engines auto-register via __init_subclass__.
Import them here to ensure registration.
"""

from app.engines.base import CountingEngineBase
from app.engines.tripwire import TripwireEngine

__all__ = [
    'CountingEngineBase',
    'TripwireEngine',
]