"""
Counting Page - Main counting display with video overlay and stats.
Optimized for RTSP streams with threaded capture.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QMessageBox
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
import cv2
import numpy as np
import math
import time

from app.ui.pages.page_base import PageBase
from app.model.data_structures import AppConfiguration
from app.model.detector import YOLODetector
from app.threads.rtsp_capture_thread import RTSPCaptureThread


class CountDisplay(QFrame):
    """Widget for displaying count statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            CountDisplay {
                background-color: #1a1a2e;
                border-radius: 10px;
                padding: 15px;
            }
            QLabel {
                color: white;
            }
        """)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("LIVE COUNT")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00d4ff;")
        layout.addWidget(title)
        
        # Entry count
        entry_frame = QFrame()
        entry_frame.setStyleSheet("background-color: #0f3d0f; border-radius: 8px; padding: 8px;")
        entry_layout = QVBoxLayout(entry_frame)
        entry_layout.setSpacing(2)
        
        entry_label = QLabel("ENTRY")
        entry_label.setAlignment(Qt.AlignCenter)
        entry_label.setStyleSheet("color: #90EE90; font-size: 11px;")
        entry_layout.addWidget(entry_label)
        
        self._entry_value = QLabel("0")
        self._entry_value.setAlignment(Qt.AlignCenter)
        self._entry_value.setFont(QFont("Arial", 24, QFont.Bold))
        self._entry_value.setStyleSheet("color: #00ff00;")
        entry_layout.addWidget(self._entry_value)
        
        layout.addWidget(entry_frame)
        
        # Exit count
        exit_frame = QFrame()
        exit_frame.setStyleSheet("background-color: #3d0f0f; border-radius: 8px; padding: 8px;")
        exit_layout = QVBoxLayout(exit_frame)
        exit_layout.setSpacing(2)
        
        exit_label = QLabel("EXIT")
        exit_label.setAlignment(Qt.AlignCenter)
        exit_label.setStyleSheet("color: #FFB6C1; font-size: 11px;")
        exit_layout.addWidget(exit_label)
        
        self._exit_value = QLabel("0")
        self._exit_value.setAlignment(Qt.AlignCenter)
        self._exit_value.setFont(QFont("Arial", 24, QFont.Bold))
        self._exit_value.setStyleSheet("color: #ff4444;")
        exit_layout.addWidget(self._exit_value)
        
        layout.addWidget(exit_frame)
        
        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background-color: #444;")
        layout.addWidget(divider)
        
        # Inside count
        inside_frame = QFrame()
        inside_frame.setStyleSheet("background-color: #1a1a3a; border-radius: 8px; padding: 8px;")
        inside_layout = QVBoxLayout(inside_frame)
        inside_layout.setSpacing(2)
        
        inside_label = QLabel("INSIDE")
        inside_label.setAlignment(Qt.AlignCenter)
        inside_label.setStyleSheet("color: #aaa; font-size: 11px;")
        inside_layout.addWidget(inside_label)
        
        self._inside_value = QLabel("0")
        self._inside_value.setAlignment(Qt.AlignCenter)
        self._inside_value.setFont(QFont("Arial", 28, QFont.Bold))
        self._inside_value.setStyleSheet("color: #ffffff;")
        inside_layout.addWidget(self._inside_value)
        
        layout.addWidget(inside_frame)
        
        layout.addStretch()
        
        # Stats display
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background-color: #2a2a3a; border-radius: 6px; padding: 6px;")
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setSpacing(2)
        
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setAlignment(Qt.AlignCenter)
        self._fps_label.setStyleSheet("color: #888; font-size: 10px;")
        stats_layout.addWidget(self._fps_label)
        
        self._resolution_label = QLabel("Res: --")
        self._resolution_label.setAlignment(Qt.AlignCenter)
        self._resolution_label.setStyleSheet("color: #888; font-size: 10px;")
        stats_layout.addWidget(self._resolution_label)
        
        layout.addWidget(stats_frame)
    
    def update_counts(self, entry: int, exit_count: int):
        """Update the displayed counts."""
        self._entry_value.setText(str(entry))
        self._exit_value.setText(str(exit_count))
        self._inside_value.setText(str(max(0, entry - exit_count)))
    
    def update_fps(self, fps: float):
        """Update FPS display."""
        self._fps_label.setText(f"FPS: {fps:.1f}")
    
    def update_resolution(self, width: int, height: int):
        """Update resolution display."""
        self._resolution_label.setText(f"Res: {width}×{height}")
    
    def reset(self):
        """Reset all counts to zero."""
        self.update_counts(0, 0)


class VideoDisplay(QLabel):
    """Widget for displaying video with overlay."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #000; border: 2px solid #333;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Starting video...")
        
        self._line_config = None
        self._frame_size = (640, 480)
    
    def set_line_config(self, line_config):
        """Set line configuration for overlay."""
        self._line_config = line_config
    
    def update_frame(self, frame, detections, counter):
        """Update display with new frame and detections."""
        if frame is None:
            return
        
        self._frame_size = (frame.shape[1], frame.shape[0])
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw tripwire line
        if self._line_config:
            p1 = (int(self._line_config.p1[0] * w), int(self._line_config.p1[1] * h))
            p2 = (int(self._line_config.p2[0] * w), int(self._line_config.p2[1] * h))
            
            # Draw line (yellow)
            cv2.line(frame, p1, p2, (0, 255, 255), 3)
            
            # Calculate perpendicular for labels
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
                
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                
                # Label positions
                offset = 50
                side_a_pos = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
                side_b_pos = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))
                
                # Determine which side is entry based on config
                entry_side = getattr(self._line_config, 'entry_direction', 'side_a')
                
                if entry_side == "side_a":
                    entry_pos, exit_pos = side_a_pos, side_b_pos
                else:
                    entry_pos, exit_pos = side_b_pos, side_a_pos
                
                # Draw labels
                cv2.putText(frame, "ENTRY", (entry_pos[0] - 35, entry_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "EXIT", (exit_pos[0] - 25, exit_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            bc_x, bc_y = int(det.bottom_center[0]), int(det.bottom_center[1])
            
            # Determine color based on side
            color = (0, 255, 0)  # Default green
            if counter and hasattr(counter, 'get_side'):
                side = counter.get_side((bc_x, bc_y))
                if side == "center":
                    color = (0, 255, 255)  # Yellow - crossing zone
                elif side == counter.entry_side:
                    color = (0, 255, 0)  # Green - entry side
                else:
                    color = (0, 0, 255)  # Red - exit side
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            if det.track_id >= 0:
                label = f"ID:{det.track_id}"
                cv2.putText(frame, label, (x1, y1 - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw bottom center point
            cv2.circle(frame, (bc_x, bc_y), 6, color, -1)
            cv2.circle(frame, (bc_x, bc_y), 6, (255, 255, 255), 2)
        
        # Convert to QPixmap and display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class SimpleCounter:
    """
    Simple tripwire counter logic that works with any line orientation.
    Uses cross product to determine which side of the line a point is on.
    """
    
    def __init__(self, line_p1: tuple, line_p2: tuple, entry_side: str = "side_a", 
                 hysteresis: float = 25, cooldown: int = 25):
        self.line_p1 = line_p1
        self.line_p2 = line_p2
        self.entry_side = entry_side
        self.hysteresis = hysteresis
        self.cooldown = cooldown
        
        self.entry_count = 0
        self.exit_count = 0
        self.track_states = {}
        
        dx = line_p2[0] - line_p1[0]
        dy = line_p2[1] - line_p1[1]
        self.line_length = math.sqrt(dx*dx + dy*dy)
    
    def _cross_product(self, point: tuple) -> float:
        x, y = point
        x1, y1 = self.line_p1
        x2, y2 = self.line_p2
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    
    def _get_distance_from_line(self, point: tuple) -> float:
        if self.line_length == 0:
            return 0
        return abs(self._cross_product(point)) / self.line_length
    
    def get_side(self, point: tuple) -> str:
        distance = self._get_distance_from_line(point)
        if distance < self.hysteresis:
            return "center"
        cross = self._cross_product(point)
        return "side_a" if cross > 0 else "side_b"
    
    def process(self, detections: list) -> list:
        events = []
        seen_ids = set()
        
        for det in detections:
            track_id = det.track_id
            if track_id < 0:
                continue
            
            seen_ids.add(track_id)
            point = det.bottom_center
            current_side = self.get_side(point)
            
            if current_side == "center":
                continue
            
            if track_id not in self.track_states:
                self.track_states[track_id] = {"side": current_side, "cooldown": 0}
                continue
            
            state = self.track_states[track_id]
            
            if state["cooldown"] > 0:
                state["cooldown"] -= 1
                continue
            
            previous_side = state["side"]
            
            if previous_side != current_side:
                if previous_side == self.entry_side:
                    self.entry_count += 1
                    events.append({"track_id": track_id, "direction": "entry"})
                else:
                    self.exit_count += 1
                    events.append({"track_id": track_id, "direction": "exit"})
                
                state["cooldown"] = self.cooldown
                state["side"] = current_side
        
        return events
    
    def reset(self):
        self.entry_count = 0
        self.exit_count = 0
        self.track_states.clear()


class CountingPage(PageBase):
    """Main counting page with video and statistics."""
    
    def __init__(self, parent=None):
        super().__init__("People Counting", parent)
        
        self._config = None
        
        # Video capture (different for RTSP vs webcam/file)
        self._cap = None  # For webcam/file
        self._rtsp_thread = None  # For RTSP
        self._is_rtsp = False
        
        self._detector = None
        self._counter = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._process_frame)
        
        self._is_running = False
        self._is_paused = False
        
        # FPS tracking for YOLO processing
        self._fps_time = time.time()
        self._fps_frames = 0
        self._process_fps = 0.0
        
        # Frame tracking for RTSP
        self._last_frame_number = -1
        
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Main horizontal layout
        main_layout = QHBoxLayout()
        
        # Video display (left side, larger)
        self._video_display = VideoDisplay()
        main_layout.addWidget(self._video_display, 3)
        
        # Stats panel (right side)
        self._count_display = CountDisplay()
        self._count_display.setMinimumWidth(160)
        self._count_display.setMaximumWidth(220)
        main_layout.addWidget(self._count_display)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.add_content(container)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self._start_btn = QPushButton("▶ Start")
        self._start_btn.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold; padding: 8px 16px;"
        )
        self._start_btn.clicked.connect(self._on_start)
        controls_layout.addWidget(self._start_btn)
        
        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._on_pause)
        controls_layout.addWidget(self._pause_btn)
        
        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        controls_layout.addWidget(self._stop_btn)
        
        controls_layout.addStretch()
        
        self._reset_btn = QPushButton("↺ Reset Counts")
        self._reset_btn.clicked.connect(self._on_reset)
        controls_layout.addWidget(self._reset_btn)
        
        self._content_layout.addLayout(controls_layout)
        
        # Status bar
        self._status_label = QLabel("Ready. Click Start to begin counting.")
        self._status_label.setStyleSheet("color: #666; padding: 5px;")
        self._content_layout.addWidget(self._status_label)
        
        # Back button only
        self.add_back_button()
        self.add_button_stretch()
    
    def set_configuration(self, config: AppConfiguration):
        """Set the application configuration."""
        self._config = config
        
        # Set line config for video overlay
        if config.roi and config.roi.lines:
            self._video_display.set_line_config(config.roi.lines[0])
        
        # Determine if RTSP
        self._is_rtsp = config.source.source_type == "rtsp"
    
    def _on_start(self):
        """Start counting."""
        if self._config is None:
            QMessageBox.warning(self, "Error", "No configuration set.")
            return
        
        if self._is_paused:
            # Resume
            self._is_paused = False
            self._timer.start(10 if self._is_rtsp else 33)
            self._start_btn.setText("▶ Start")
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._status_label.setText("Counting resumed...")
            return
        
        self._status_label.setText("Loading model...")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Initialize detector
        self._detector = YOLODetector(self._config.model)
        if not self._detector.initialize():
            QMessageBox.critical(self, "Error", "Failed to load YOLO model.")
            return
        
        # Get resolution for counter setup
        width, height = 1280, 720  # Default for RTSP
        
        if self._is_rtsp:
            # === RTSP MODE: Use threaded capture ===
            self._status_label.setText("Connecting to RTSP stream (720p mode)...")
            QApplication.processEvents()
            
            self._rtsp_thread = RTSPCaptureThread(
                rtsp_url=self._config.source.path,
                target_width=1280,
                target_height=720
            )
            
            if not self._rtsp_thread.start():
                QMessageBox.critical(
                    self, "Error", 
                    f"Failed to connect to RTSP stream:\n{self._rtsp_thread.error_message}"
                )
                return
            
            # Wait for first frame
            time.sleep(0.5)
            
            width, height = self._rtsp_thread.target_size
            original_size = self._rtsp_thread.original_size
            
            self._status_label.setText(
                f"RTSP connected: {original_size[0]}×{original_size[1]} → {width}×{height}"
            )
            
        else:
            # === WEBCAM/FILE MODE: Use direct capture (unchanged) ===
            self._status_label.setText("Connecting to video source...")
            QApplication.processEvents()
            
            if self._config.source.source_type == "video_file":
                self._cap = cv2.VideoCapture(self._config.source.path)
            else:
                self._cap = cv2.VideoCapture(self._config.source.device_id)
            
            if not self._cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video source.")
                return
            
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize counter with line from config
        if self._config.roi and self._config.roi.lines:
            line = self._config.roi.lines[0]
            p1 = (line.p1[0] * width, line.p1[1] * height)
            p2 = (line.p2[0] * width, line.p2[1] * height)
            entry_side = line.entry_direction
            
            self._counter = SimpleCounter(
                line_p1=p1,
                line_p2=p2,
                entry_side=entry_side,
                hysteresis=25,
                cooldown=25
            )
        else:
            self._counter = SimpleCounter(
                line_p1=(width // 2, 0),
                line_p2=(width // 2, height),
                entry_side="side_a"
            )
        
        # Update resolution display
        self._count_display.update_resolution(width, height)
        
        # Update UI
        self._is_running = True
        self._is_paused = False
        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        
        self._fps_time = time.time()
        self._fps_frames = 0
        self._last_frame_number = -1
        
        self._status_label.setText("Counting in progress...")
        
        # Start processing timer
        # RTSP: faster timer since we're just checking for new frames
        # Webcam/File: 30fps is enough
        self._timer.start(10 if self._is_rtsp else 33)
    
    def _on_pause(self):
        """Pause counting."""
        self._is_paused = True
        self._timer.stop()
        
        self._start_btn.setText("▶ Resume")
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        
        self._status_label.setText("Paused")
    
    def _on_stop(self):
        """Stop counting."""
        self.stop()
        self._status_label.setText("Stopped. Click Start to begin again.")
    
    def _on_reset(self):
        """Reset counts."""
        if self._counter:
            self._counter.reset()
        self._count_display.reset()
        
        if self._detector:
            self._detector.reset_tracker()
    
    def _process_frame(self):
        """Process a single frame."""
        if not self._is_running:
            return
        
        frame = None
        
        if self._is_rtsp:
            # === RTSP MODE ===
            if self._rtsp_thread is None:
                return
            
            result = self._rtsp_thread.get_frame()
            if result is None:
                if not self._rtsp_thread.is_connected:
                    self._status_label.setText("RTSP disconnected! Reconnecting...")
                return
            
            frame, frame_number = result
            
            # Skip if same frame (already processed)
            if frame_number == self._last_frame_number:
                return
            
            self._last_frame_number = frame_number
            
            # Update FPS from capture thread
            self._count_display.update_fps(self._rtsp_thread.fps)
            
        else:
            # === WEBCAM/FILE MODE (unchanged) ===
            if self._cap is None:
                return
            
            ret, frame = self._cap.read()
            
            if not ret:
                if self._config.source.source_type == "video_file":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return
                else:
                    self._status_label.setText("Video source disconnected!")
                    return
        
        if frame is None:
            return
        
        # Run detection
        detections = self._detector.track(frame)
        
        # Run counting
        events = self._counter.process(detections)
        
        # Log events
        for event in events:
            direction = event["direction"].upper()
            track_id = event["track_id"]
            print(f"  [{direction}] Track {track_id}")
        
        # Update displays
        self._video_display.update_frame(frame, detections, self._counter)
        
        self._count_display.update_counts(
            self._counter.entry_count,
            self._counter.exit_count
        )
        
        # Update processing FPS (for webcam/file mode)
        if not self._is_rtsp:
            self._fps_frames += 1
            if self._fps_frames >= 20:
                elapsed = time.time() - self._fps_time
                if elapsed > 0:
                    self._process_fps = self._fps_frames / elapsed
                self._fps_time = time.time()
                self._fps_frames = 0
                self._count_display.update_fps(self._process_fps)
    
    def stop(self):
        """Stop all processing."""
        self._is_running = False
        self._is_paused = False
        self._timer.stop()
        
        # Stop RTSP thread
        if self._rtsp_thread:
            self._rtsp_thread.stop()
            self._rtsp_thread = None
        
        # Release webcam/file capture
        if self._cap:
            self._cap.release()
            self._cap = None
        
        self._detector = None
        
        # Reset UI
        self._start_btn.setText("▶ Start")
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
    
    def hideEvent(self, event):
        """Handle page hidden."""
        super().hideEvent(event)
        self.stop()