"""
RTSP Capture Thread - Optimized for RTSP streams.
Captures frames in background, smart resize, and keeps only the latest frame.
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
    - Smart resize: only downscale if input > target (never upscale)
    - Only keeps the latest frame (no buffer buildup)
    - Non-blocking frame retrieval
    """
    
    def __init__(
        self, 
        rtsp_url: str, 
        max_width: int = 1280,   # Maximum width (downscale if larger)
        max_height: int = 720    # Maximum height (downscale if larger)
    ):
        """
        Initialize RTSP capture.
        
        Args:
            rtsp_url: RTSP stream URL
            max_width: Maximum width - downscale if input is larger (default 1280)
            max_height: Maximum height - downscale if input is larger (default 720)
        """
        self._url = rtsp_url
        self._max_width = max_width
        self._max_height = max_height
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[Thread] = None
        self._lock = Lock()
        
        self._running = False
        self._frame: Optional[np.ndarray] = None
        self._frame_number = 0
        
        # Sizes
        self._original_size: Tuple[int, int] = (0, 0)
        self._output_size: Tuple[int, int] = (0, 0)
        self._needs_resize = False
        
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
        """Original input resolution from camera."""
        return self._original_size
    
    @property
    def output_size(self) -> Tuple[int, int]:
        """Actual output resolution (after smart resize)."""
        return self._output_size
    
    @property
    def target_size(self) -> Tuple[int, int]:
        """Alias for output_size (backward compatibility)."""
        return self._output_size
    
    @property
    def is_resizing(self) -> bool:
        """Whether frames are being downscaled."""
        return self._needs_resize
    
    @property
    def error_message(self) -> str:
        return self._error_message
    
    def _calculate_output_size(self, input_width: int, input_height: int) -> Tuple[int, int, bool]:
        """
        Calculate output size using smart resize logic.
        Only downscale if input is larger than max dimensions.
        Maintains aspect ratio.
        
        Returns:
            (output_width, output_height, needs_resize)
        """
        # If input is smaller or equal to max, keep original
        if input_width <= self._max_width and input_height <= self._max_height:
            return (input_width, input_height, False)
        
        # Calculate scale factor to fit within max dimensions (maintain aspect ratio)
        scale_w = self._max_width / input_width
        scale_h = self._max_height / input_height
        scale = min(scale_w, scale_h)
        
        output_width = int(input_width * scale)
        output_height = int(input_height * scale)
        
        # Ensure even dimensions (some codecs require this)
        output_width = output_width - (output_width % 2)
        output_height = output_height - (output_height % 2)
        
        return (output_width, output_height, True)
    
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
        
        # Calculate smart output size
        out_w, out_h, needs_resize = self._calculate_output_size(
            self._original_size[0], 
            self._original_size[1]
        )
        self._output_size = (out_w, out_h)
        self._needs_resize = needs_resize
        
        # Log the decision
        if needs_resize:
            print(f"[RTSP] Input: {self._original_size[0]}×{self._original_size[1]} → "
                  f"Downscaling to: {out_w}×{out_h}")
        else:
            print(f"[RTSP] Input: {self._original_size[0]}×{self._original_size[1]} → "
                  f"Keeping original (no resize needed)")
        
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
            
            # Smart resize: only downscale if needed
            if self._needs_resize:
                frame = cv2.resize(
                    frame, 
                    self._output_size,
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
                return (self._frame.copy(), self._frame_number)
        return None
    
    def stop(self):
        """Stop the capture thread."""
        self._running = False
        
        if hasattr(self, '_thread') and self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        
        if hasattr(self, '_cap') and self._cap:
            self._cap.release()
            self._cap = None
        
        if hasattr(self, '_frame'):
            self._frame = None
        
        self._connected = False
    
    def get_info(self) -> dict:
        """Get capture information."""
        return {
            'url': self._url,
            'original_size': self._original_size,
            'output_size': self._output_size,
            'is_resizing': self._needs_resize,
            'max_size': (self._max_width, self._max_height),
            'fps': self._fps,
            'connected': self._connected
        }
    
    def __del__(self):
        try:
            self.stop()
        except AttributeError:
            # Object wasn't fully initialized
            pass