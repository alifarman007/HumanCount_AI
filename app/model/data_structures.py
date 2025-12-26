"""
Core data structures for the People Counting system.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Deque, List, Any
from collections import deque
from enum import Enum, auto
import time


class CountDirection(Enum):
    """Direction of counted movement"""
    ENTRY = auto()
    EXIT = auto()
    SPAWN = auto()      # Appeared inside (car passenger)
    VANISH = auto()     # Disappeared inside (got into car)


class ZoneState(Enum):
    """State for zone-based tracking"""
    OUTSIDE = auto()
    ZONE_A = auto()
    ZONE_B = auto()
    INSIDE = auto()


class AlertLevel(Enum):
    """Alert levels for occupancy"""
    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()
    EXCEEDED = auto()


@dataclass
class DebounceConfig:
    """Unified debounce settings for all engines"""
    
    # Spatial hysteresis (pixels)
    hysteresis_distance: float = 10.0
    
    # Temporal debounce (frames)
    min_frames_stable: int = 2
    
    # Cooldown after count (frames)
    cooldown_frames: int = 15
    
    # Zone-specific (for state machine)
    min_frames_in_zone: int = 3
    
    # Vanish detection (seconds)
    vanish_timeout_seconds: float = 3.0
    
    def to_dict(self) -> Dict:
        return {
            'hysteresis_distance': self.hysteresis_distance,
            'min_frames_stable': self.min_frames_stable,
            'cooldown_frames': self.cooldown_frames,
            'min_frames_in_zone': self.min_frames_in_zone,
            'vanish_timeout_seconds': self.vanish_timeout_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DebounceConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrackState:
    """State for a single tracked object"""
    
    track_id: int
    
    # Position tracking (circular buffer for efficiency)
    position_history: Deque[tuple] = field(
        default_factory=lambda: deque(maxlen=30)  # ~1 second at 30fps
    )
    
    # Timestamps
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    
    # Counting flags (prevents double-counting)
    counted_entry: bool = False
    counted_exit: bool = False
    counted_entry_gate: bool = False
    counted_exit_gate: bool = False
    counted_spawn: bool = False
    counted_vanish: bool = False
    
    # Zone state machine
    zone_state: ZoneState = ZoneState.OUTSIDE
    zone_state_frames: int = 0  # Frames in current state (for debounce)
    
    # Spawn/Vanish detection
    spawned_inside: bool = False  # First detection was inside a zone
    
    # Last known side of line (for tripwire): -1 (right), 0 (on), 1 (left)
    last_line_side: Optional[int] = None
    
    # For multi-segment: per-segment side tracking
    segment_sides: Dict[int, int] = field(default_factory=dict)
    
    # Cooldown timer
    cooldown_remaining: int = 0
    
    @property
    def current_position(self) -> Optional[tuple]:
        """Get most recent position"""
        return self.position_history[-1] if self.position_history else None
    
    @property
    def previous_position(self) -> Optional[tuple]:
        """Get second-most recent position"""
        if len(self.position_history) >= 2:
            return self.position_history[-2]
        return None
    
    @property
    def movement_vector(self) -> Optional[tuple]:
        """Calculate movement direction"""
        curr = self.current_position
        prev = self.previous_position
        if curr and prev:
            return (curr[0] - prev[0], curr[1] - prev[1])
        return None
    
    def add_position(self, position: tuple, frame_number: int):
        """Add new position to history"""
        self.position_history.append(position)
        self.last_seen_frame = frame_number
        if self.first_seen_frame == 0:
            self.first_seen_frame = frame_number


@dataclass
class CountEvent:
    """Immutable event representing a count"""
    
    track_id: int
    direction: CountDirection
    engine_type: str
    timestamp: float = field(default_factory=time.time)
    position: tuple = (0, 0)
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'track_id': self.track_id,
            'direction': self.direction.name,
            'engine_type': self.engine_type,
            'timestamp': self.timestamp,
            'position': self.position,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class Detection:
    """Single detection from YOLO"""
    
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # 0 = person
    
    @property
    def center(self) -> tuple:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def bottom_center(self) -> tuple:
        """Get bottom center point (feet position)"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


@dataclass 
class LineConfig:
    """Configuration for a line (tripwire)"""
    
    id: str
    p1: tuple  # (x, y) normalized 0-1
    p2: tuple  # (x, y) normalized 0-1
    entry_direction: str = "left_to_right"  # or "right_to_left", "top_to_bottom", "bottom_to_top"
    label: str = ""
    
    def to_pixel_coords(self, width: int, height: int) -> tuple:
        """Convert normalized coords to pixel coords"""
        return (
            (int(self.p1[0] * width), int(self.p1[1] * height)),
            (int(self.p2[0] * width), int(self.p2[1] * height))
        )
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'p1': self.p1,
            'p2': self.p2,
            'entry_direction': self.entry_direction,
            'label': self.label
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LineConfig':
        return cls(
            id=data['id'],
            p1=tuple(data['p1']),
            p2=tuple(data['p2']),
            entry_direction=data.get('entry_direction', 'left_to_right'),
            label=data.get('label', '')
        )


@dataclass
class PolygonConfig:
    """Configuration for a polygon zone"""
    
    id: str
    vertices: List[tuple]  # List of (x, y) normalized 0-1
    label: str = ""
    zone_type: str = "detection"  # "detection", "confirmation", "parking", "occupancy"
    
    def to_pixel_coords(self, width: int, height: int) -> List[tuple]:
        """Convert normalized coords to pixel coords"""
        return [(int(v[0] * width), int(v[1] * height)) for v in self.vertices]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'vertices': self.vertices,
            'label': self.label,
            'zone_type': self.zone_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PolygonConfig':
        return cls(
            id=data['id'],
            vertices=[tuple(v) for v in data['vertices']],
            label=data.get('label', ''),
            zone_type=data.get('zone_type', 'detection')
        )


@dataclass
class ROIConfiguration:
    """Complete ROI configuration for a module"""
    
    module_type: str
    lines: List[LineConfig] = field(default_factory=list)
    polygons: List[PolygonConfig] = field(default_factory=list)
    debounce: DebounceConfig = field(default_factory=DebounceConfig)
    module_specific: Dict = field(default_factory=dict)  # Capacity, etc.
    
    def to_dict(self) -> Dict:
        return {
            'module_type': self.module_type,
            'lines': [l.to_dict() for l in self.lines],
            'polygons': [p.to_dict() for p in self.polygons],
            'debounce': self.debounce.to_dict(),
            'module_specific': self.module_specific
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ROIConfiguration':
        return cls(
            module_type=data['module_type'],
            lines=[LineConfig.from_dict(l) for l in data.get('lines', [])],
            polygons=[PolygonConfig.from_dict(p) for p in data.get('polygons', [])],
            debounce=DebounceConfig.from_dict(data.get('debounce', {})),
            module_specific=data.get('module_specific', {})
        )


@dataclass
class SourceConfig:
    """Video source configuration"""
    
    source_type: str  # "video_file", "rtsp", "webcam"
    path: str = ""  # File path or RTSP URL
    device_id: int = 0  # For webcam
    resolution: tuple = (1920, 1080)
    fps: int = 30
    
    def to_dict(self) -> Dict:
        return {
            'source_type': self.source_type,
            'path': self.path,
            'device_id': self.device_id,
            'resolution': self.resolution,
            'fps': self.fps
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SourceConfig':
        return cls(
            source_type=data['source_type'],
            path=data.get('path', ''),
            device_id=data.get('device_id', 0),
            resolution=tuple(data.get('resolution', (1920, 1080))),
            fps=data.get('fps', 30)
        )


@dataclass
class ModelConfig:
    """YOLO model configuration"""
    
    mode: str = "cpu"  # "cpu" or "gpu"
    device_id: int = 0  # GPU device ID
    model_size: str = "n"  # n, s, m, l, x
    confidence: float = 0.5
    
    @property
    def model_name(self) -> str:
        """Get full model name"""
        return f"yolo11{self.model_size}.pt"
    
    def to_dict(self) -> Dict:
        return {
            'mode': self.mode,
            'device_id': self.device_id,
            'model_size': self.model_size,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConfig':
        return cls(
            mode=data.get('mode', 'cpu'),
            device_id=data.get('device_id', 0),
            model_size=data.get('model_size', 'n'),
            confidence=data.get('confidence', 0.5)
        )


@dataclass
class AppConfiguration:
    """Complete application configuration (for save/load)"""
    
    version: str = "1.0.0"
    module_type: str = ""
    source: SourceConfig = field(default_factory=lambda: SourceConfig(source_type="webcam"))
    model: ModelConfig = field(default_factory=ModelConfig)
    roi: Optional[ROIConfiguration] = None
    created_at: str = ""
    last_modified: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'version': self.version,
            'module_type': self.module_type,
            'source': self.source.to_dict(),
            'model': self.model.to_dict(),
            'roi': self.roi.to_dict() if self.roi else None,
            'created_at': self.created_at,
            'last_modified': self.last_modified
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AppConfiguration':
        return cls(
            version=data.get('version', '1.0.0'),
            module_type=data.get('module_type', ''),
            source=SourceConfig.from_dict(data.get('source', {'source_type': 'webcam'})),
            model=ModelConfig.from_dict(data.get('model', {})),
            roi=ROIConfiguration.from_dict(data['roi']) if data.get('roi') else None,
            created_at=data.get('created_at', ''),
            last_modified=data.get('last_modified', '')
        )