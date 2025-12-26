"""
Video Source - Handles video input from files, RTSP streams, and webcams.
Runs in separate thread for smooth playback.
"""

import cv2
import time
from typing import Optional, Tuple
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from enum import Enum, auto


class SourceType(Enum):
    VIDEO_FILE = auto()
    RTSP = auto()
    WEBCAM = auto()


class VideoSource(QThread):
    """
    Video capture thread for continuous frame reading.
    
    Signals:
        frame_ready: Emits (frame, frame_number) when new frame available
        source_error: Emits error message on failure
        source_ended: Emits when video file ends
        fps_updated: Emits current FPS
    """
    
    frame_ready = Signal(object, int)  # frame (numpy array), frame_number
    source_error = Signal(str)
    source_ended = Signal()
    fps_updated = Signal(float)
    resolution_detected = Signal(int, int)  # width, height
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._source_type: SourceType = SourceType.WEBCAM
        self._source_path: str = ""
        self._device_id: int = 0
        
        # State
        self._running = False
        self._paused = False
        self._frame_count = 0
        
        # Synchronization
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()
        
        # FPS tracking
        self._target_fps = 30
        self._actual_fps = 0.0
        self._fps_update_interval = 30  # Update FPS every 30 frames
        self._last_fps_time = 0.0
        self._fps_frame_count = 0
        
        # Resolution
        self._width = 0
        self._height = 0
        
        # Reconnection for RTSP
        self._reconnect_attempts = 3
        self._reconnect_delay = 2.0  # seconds
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def is_paused(self) -> bool:
        return self._paused
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)
    
    @property
    def fps(self) -> float:
        return self._actual_fps if self._actual_fps > 0 else self._target_fps
    
    def set_source_file(self, path: str):
        """Set video file as source."""
        self._source_type = SourceType.VIDEO_FILE
        self._source_path = path
    
    def set_source_rtsp(self, url: str):
        """Set RTSP stream as source."""
        self._source_type = SourceType.RTSP
        self._source_path = url
    
    def set_source_webcam(self, device_id: int = 0):
        """Set webcam as source."""
        self._source_type = SourceType.WEBCAM
        self._device_id = device_id
    
    def _open_source(self) -> bool:
        """Open the video source."""
        try:
            if self._source_type == SourceType.VIDEO_FILE:
                self._cap = cv2.VideoCapture(self._source_path)
            elif self._source_type == SourceType.RTSP:
                # RTSP with buffering settings for lower latency
                self._cap = cv2.VideoCapture(self._source_path, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:  # WEBCAM
                self._cap = cv2.VideoCapture(self._device_id)
            
            if not self._cap.isOpened():
                self.source_error.emit(f"Failed to open source: {self._source_path or self._device_id}")
                return False
            
            # Get properties
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._target_fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            if self._target_fps <= 0:
                self._target_fps = 30  # Default fallback
            
            self.resolution_detected.emit(self._width, self._height)
            
            return True
            
        except Exception as e:
            self.source_error.emit(f"Error opening source: {str(e)}")
            return False
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to RTSP stream."""
        if self._source_type != SourceType.RTSP:
            return False
        
        for attempt in range(self._reconnect_attempts):
            self.source_error.emit(f"Reconnecting... attempt {attempt + 1}/{self._reconnect_attempts}")
            time.sleep(self._reconnect_delay)
            
            if self._cap:
                self._cap.release()
            
            if self._open_source():
                return True
        
        return False
    
    def run(self):
        """Main capture loop."""
        if not self._open_source():
            return
        
        self._running = True
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        
        frame_time = 1.0 / self._target_fps
        
        while self._running:
            # Handle pause
            self._mutex.lock()
            while self._paused and self._running:
                self._pause_condition.wait(self._mutex)
            self._mutex.unlock()
            
            if not self._running:
                break
            
            loop_start = time.time()
            
            # Read frame
            ret, frame = self._cap.read()
            
            if not ret:
                if self._source_type == SourceType.VIDEO_FILE:
                    # Video ended
                    self.source_ended.emit()
                    break
                elif self._source_type == SourceType.RTSP:
                    # Try reconnect
                    if not self._reconnect():
                        self.source_error.emit("Failed to reconnect to RTSP stream")
                        break
                    continue
                else:
                    # Webcam error
                    self.source_error.emit("Failed to read from webcam")
                    break
            
            self._frame_count += 1
            
            # Emit frame
            self.frame_ready.emit(frame, self._frame_count)
            
            # Update FPS
            self._fps_frame_count += 1
            if self._fps_frame_count >= self._fps_update_interval:
                current_time = time.time()
                elapsed = current_time - self._last_fps_time
                if elapsed > 0:
                    self._actual_fps = self._fps_frame_count / elapsed
                    self.fps_updated.emit(self._actual_fps)
                self._last_fps_time = current_time
                self._fps_frame_count = 0
            
            # Frame rate control (for video files)
            if self._source_type == SourceType.VIDEO_FILE:
                elapsed = time.time() - loop_start
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        # Cleanup
        if self._cap:
            self._cap.release()
            self._cap = None
        
        self._running = False
    
    def pause(self):
        """Pause video capture."""
        self._mutex.lock()
        self._paused = True
        self._mutex.unlock()
    
    def resume(self):
        """Resume video capture."""
        self._mutex.lock()
        self._paused = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
    
    def stop(self):
        """Stop video capture."""
        self._mutex.lock()
        self._running = False
        self._paused = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        
        # Wait for thread to finish
        self.wait(5000)  # 5 second timeout
    
    def seek(self, frame_number: int):
        """Seek to specific frame (video files only)."""
        if self._source_type == SourceType.VIDEO_FILE and self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._frame_count = frame_number
    
    def get_frame_sync(self) -> Optional[Tuple]:
        """
        Get single frame synchronously (for ROI configuration).
        Returns (frame, frame_number) or None.
        """
        if not self._cap or not self._cap.isOpened():
            if not self._open_source():
                return None
        
        ret, frame = self._cap.read()
        if ret:
            return (frame, 0)
        return None


def enumerate_webcams(max_devices: int = 5) -> list:
    """
    Find available webcam devices.
    
    Returns:
        List of (device_id, name) tuples
    """
    devices = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            devices.append((i, f"Camera {i}"))
            cap.release()
    return devices