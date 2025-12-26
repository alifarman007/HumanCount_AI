"""
Track Manager - Central state management for all tracked objects.
Maintains position history, counting flags, and zone states per track.
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict

from app.model.data_structures import (
    TrackState, 
    Detection, 
    ZoneState,
    DebounceConfig
)


class TrackManager:
    """
    Manages state for all tracked objects across frames.
    
    Responsibilities:
    - Maintain position history for each track
    - Track counting flags to prevent double-counting
    - Clean up stale tracks
    - Provide immutable snapshots to counting engines
    """
    
    def __init__(
        self, 
        history_length: int = 30,
        stale_threshold: int = 90,
        position_type: str = "bottom_center"
    ):
        """
        Initialize Track Manager.
        
        Args:
            history_length: Max positions to keep per track (default: 30 = ~1 sec at 30fps)
            stale_threshold: Frames before removing lost tracks (default: 90 = 3 sec)
            position_type: How to extract position from bbox ("center" or "bottom_center")
        """
        self._tracks: Dict[int, TrackState] = {}
        self._history_length = history_length
        self._stale_threshold = stale_threshold
        self._position_type = position_type
        self._frame_count = 0
        
        # Track IDs seen in current frame
        self._current_frame_ids: Set[int] = set()
        
        # Track IDs that were just created this frame (for spawn detection)
        self._new_track_ids: Set[int] = set()
        
        # Track IDs that vanished (for vanish detection)
        self._vanished_ids: Dict[int, int] = {}  # track_id -> frames_since_seen
    
    @property
    def frame_count(self) -> int:
        """Current frame number"""
        return self._frame_count
    
    @property
    def active_track_count(self) -> int:
        """Number of currently active tracks"""
        return len(self._tracks)
    
    @property
    def new_track_ids(self) -> Set[int]:
        """Track IDs that were just created this frame"""
        return self._new_track_ids.copy()
    
    def update(self, detections: List[Detection]) -> Dict[int, TrackState]:
        """
        Update track states with new detections.
        
        Args:
            detections: List of Detection objects from YOLO+tracker
        
        Returns:
            Dictionary of track_id -> TrackState (copy for safety)
        """
        self._frame_count += 1
        self._new_track_ids.clear()
        self._current_frame_ids.clear()
        
        for det in detections:
            track_id = det.track_id
            position = self._get_position(det)
            self._current_frame_ids.add(track_id)
            
            if track_id not in self._tracks:
                # New track
                self._tracks[track_id] = TrackState(track_id=track_id)
                self._new_track_ids.add(track_id)
                
                # Remove from vanished if it reappeared
                if track_id in self._vanished_ids:
                    del self._vanished_ids[track_id]
            
            # Update track state
            track = self._tracks[track_id]
            track.add_position(position, self._frame_count)
            
            # Decrease cooldown timer
            if track.cooldown_remaining > 0:
                track.cooldown_remaining -= 1
        
        # Update vanished tracking
        self._update_vanished_tracks()
        
        # Clean up stale tracks
        self._cleanup_stale_tracks()
        
        return self.get_all_tracks()
    
    def _get_position(self, detection: Detection) -> tuple:
        """Extract position from detection based on position_type setting."""
        if self._position_type == "bottom_center":
            return detection.bottom_center
        return detection.center
    
    def _update_vanished_tracks(self):
        """Track how long each track has been missing."""
        for track_id in list(self._tracks.keys()):
            if track_id not in self._current_frame_ids:
                # Track not seen this frame
                if track_id not in self._vanished_ids:
                    self._vanished_ids[track_id] = 0
                self._vanished_ids[track_id] += 1
    
    def _cleanup_stale_tracks(self):
        """Remove tracks not seen for stale_threshold frames."""
        stale_ids = [
            track_id for track_id, track in self._tracks.items()
            if self._frame_count - track.last_seen_frame > self._stale_threshold
        ]
        
        for track_id in stale_ids:
            del self._tracks[track_id]
            if track_id in self._vanished_ids:
                del self._vanished_ids[track_id]
    
    def get_all_tracks(self) -> Dict[int, TrackState]:
        """Get copy of all track states."""
        return self._tracks.copy()
    
    def get_track(self, track_id: int) -> Optional[TrackState]:
        """Get single track state by ID."""
        return self._tracks.get(track_id)
    
    def get_active_positions(self) -> Dict[int, tuple]:
        """
        Get current positions of all active tracks.
        
        Returns:
            Dict of track_id -> (x, y) position
        """
        positions = {}
        for track_id, track in self._tracks.items():
            if track.current_position:
                positions[track_id] = track.current_position
        return positions
    
    def is_new_track(self, track_id: int) -> bool:
        """Check if track was just created this frame."""
        return track_id in self._new_track_ids
    
    def get_vanished_frames(self, track_id: int) -> int:
        """
        Get how many frames a track has been missing.
        
        Returns:
            Number of frames since last seen, or 0 if currently visible
        """
        return self._vanished_ids.get(track_id, 0)
    
    def is_vanished(self, track_id: int, threshold_frames: int) -> bool:
        """
        Check if track has been missing for at least threshold_frames.
        
        Args:
            track_id: Track to check
            threshold_frames: Minimum frames missing to be considered vanished
        
        Returns:
            True if track has vanished (missing >= threshold)
        """
        return self._vanished_ids.get(track_id, 0) >= threshold_frames
    
    # === Flag Update Methods ===
    
    def set_counted_entry(self, track_id: int, value: bool = True):
        """Mark track as counted for entry."""
        if track_id in self._tracks:
            self._tracks[track_id].counted_entry = value
    
    def set_counted_exit(self, track_id: int, value: bool = True):
        """Mark track as counted for exit."""
        if track_id in self._tracks:
            self._tracks[track_id].counted_exit = value
    
    def set_counted_entry_gate(self, track_id: int, value: bool = True):
        """Mark track as counted for entry gate."""
        if track_id in self._tracks:
            self._tracks[track_id].counted_entry_gate = value
    
    def set_counted_exit_gate(self, track_id: int, value: bool = True):
        """Mark track as counted for exit gate."""
        if track_id in self._tracks:
            self._tracks[track_id].counted_exit_gate = value
    
    def set_counted_spawn(self, track_id: int, value: bool = True):
        """Mark track as counted for spawn."""
        if track_id in self._tracks:
            self._tracks[track_id].counted_spawn = value
            self._tracks[track_id].spawned_inside = value
    
    def set_counted_vanish(self, track_id: int, value: bool = True):
        """Mark track as counted for vanish."""
        if track_id in self._tracks:
            self._tracks[track_id].counted_vanish = value
    
    def set_cooldown(self, track_id: int, frames: int):
        """Set cooldown timer for a track."""
        if track_id in self._tracks:
            self._tracks[track_id].cooldown_remaining = frames
    
    def is_in_cooldown(self, track_id: int) -> bool:
        """Check if track is in cooldown period."""
        if track_id in self._tracks:
            return self._tracks[track_id].cooldown_remaining > 0
        return False
    
    def set_line_side(self, track_id: int, side: int):
        """Update last known line side for a track."""
        if track_id in self._tracks:
            self._tracks[track_id].last_line_side = side
    
    def set_segment_side(self, track_id: int, segment_idx: int, side: int):
        """Update line side for specific segment (polyline)."""
        if track_id in self._tracks:
            self._tracks[track_id].segment_sides[segment_idx] = side
    
    def set_zone_state(self, track_id: int, state: ZoneState):
        """Update zone state for a track."""
        if track_id in self._tracks:
            track = self._tracks[track_id]
            if track.zone_state != state:
                track.zone_state = state
                track.zone_state_frames = 0
            else:
                track.zone_state_frames += 1
    
    def reset(self):
        """Reset all tracking state."""
        self._tracks.clear()
        self._vanished_ids.clear()
        self._new_track_ids.clear()
        self._current_frame_ids.clear()
        self._frame_count = 0