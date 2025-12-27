"""
RTSP Capture Thread - Optimized for RTSP streams.
Captures frames in background, resizes, and keeps only the latest frame.
"""

import cv2
import time
from threading import Thread, Lock
from typing import Optional, Tuple
import numpy as np


class RTSPCaptureThread:
    """
    Threaded RTSP capture that always provides the latest frame.
    
    Features:
    - Background thread continuously reads frames
    - Resizes to target resolution for faster processing
    - Only keeps the latest frame (no buffer buildup)
    - Non-blocking frame retrieval
    """
    
    def __init__(
        self, 
        rtsp_url: str, 
        target_width: int = 1280,  # 720p width
        target_height: int = 720   # 720p height
    ):
        """
        Initialize RTSP capture.
        
        Args:
            rtsp_url: RTSP stream URL
            target_width: Target width for resize (default 1280 for 720p)
            target_height: Target height for resize (default 720 for 720p)
        """
        self._url = rtsp_url
        self._target_width = target_width
        self._target_height = target_height
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[Thread] = None
        self._lock = Lock()
        
        self._running = False
        self._frame: Optional[np.ndarray] = None
        self._frame_number = 0
        self._original_size: Tuple[int, int] = (0, 0)
        
        # Stats
        self._fps = 0.0
        self._frames_captured = 0
        self._last_fps_time = time.time()
        self._connected = False
        self._error_message = ""
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def original_size(self) -> Tuple[int, int]:
        return self._original_size
    
    @property
    def target_size(self) -> Tuple[int, int]:
        return (self._target_width, self._target_height)
    
    @property
    def error_message(self) -> str:
        return self._error_message
    
    def start(self) -> bool:
        """
        Start the capture thread.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
        
        # Open RTSP with optimized settings
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        
        if not self._cap.isOpened():
            self._error_message = "Failed to connect to RTSP stream"
            return False
        
        # Set buffer size to minimum (reduces latency)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get original resolution
        self._original_size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
        # Start capture thread
        self._running = True
        self._connected = True
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def _capture_loop(self):
        """Background capture loop."""
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                # Try to reconnect
                if reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    print(f"[RTSP] Reconnecting... attempt {reconnect_attempts}")
                    time.sleep(2)
                    self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    continue
                else:
                    self._error_message = "Failed to reconnect to RTSP stream"
                    self._connected = False
                    break
            
            ret, frame = self._cap.read()
            
            if not ret:
                self._connected = False
                continue
            
            # Reset reconnect counter on successful read
            reconnect_attempts = 0
            self._connected = True
            
            # Resize frame to target resolution
            if frame.shape[1] != self._target_width or frame.shape[0] != self._target_height:
                frame = cv2.resize(
                    frame, 
                    (self._target_width, self._target_height),
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Update latest frame (thread-safe)
            with self._lock:
                self._frame = frame
                self._frame_number += 1
            
            # Update FPS
            self._frames_captured += 1
            if self._frames_captured >= 30:
                now = time.time()
                elapsed = now - self._last_fps_time
                if elapsed > 0:
                    self._fps = self._frames_captured / elapsed
                self._last_fps_time = now
                self._frames_captured = 0
        
        # Cleanup
        if self._cap:
            self._cap.release()
            self._cap = None
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Get the latest frame (non-blocking).
        
        Returns:
            (frame, frame_number) or None if no frame available
        """
        with self._lock:
            if self._frame is not None:
                # Return a copy to avoid race conditions
                return (self._frame.copy(), self._frame_number)
        return None
    
    def stop(self):
        """Stop the capture thread."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        self._frame = None
        self._connected = False
    
    def __del__(self):
        self.stop()