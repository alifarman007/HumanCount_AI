"""
Counting Page - Live video with people counting.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy, QApplication
)
from PySide6.QtCore import Signal, Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap, QFont
import cv2
import numpy as np
import math
import time

from app.ui.pages.page_base import PageBase
from app.model.data_structures import SourceConfig, ModelConfig, ROIConfiguration


class CountingPage(PageBase):
    """Page for live counting display."""
    
    counting_stopped = Signal()
    
    def __init__(self, parent=None):
        super().__init__("People Counting", parent)
        
        self._source_config = None
        self._model_config = None
        self._roi_config = None
        
        # Video capture
        self._cap = None
        self._rtsp_thread = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._process_frame)
        
        # YOLO model
        self._model = None
        
        # Counting state
        self._entry_count = 0
        self._exit_count = 0
        
        # Track states: {track_id: {'side': str, 'cooldown': int}}
        self._track_states = {}
        
        # For polyline: {track_id: {segment_idx: {'side': str, 'cooldown': int}}}
        self._track_segment_states = {}
        
        # Cooldown frames after a count (prevents double-counting)
        self._cooldown_frames = 30
        
        # Line geometry (pixels)
        self._line_p1 = None
        self._line_p2 = None
        self._line_length = 0
        self._entry_side = "side_a"
        
        # Polyline geometry (pixels) - list of (p1, p2) tuples
        self._polyline_segments = []
        
        # Polygon geometry (pixels)
        self._polygon_vertices = []
        
        # Frame info
        self._frame_size = (640, 480)
        self._fps = 30
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        self._is_running = False
        self._is_paused = False
        
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        main_layout = QHBoxLayout()
        
        # Video display
        self._video_label = QLabel()
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setText("Click Start to begin counting...")
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self._video_label, stretch=3)
        
        # Stats panel
        stats_panel = QFrame()
        stats_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        stats_panel.setMinimumWidth(180)
        stats_panel.setMaximumWidth(250)
        stats_layout = QVBoxLayout(stats_panel)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("LIVE COUNT")
        title.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold; border: 1px solid #00ff00; padding: 5px;")
        title.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(title)
        
        stats_layout.addSpacing(20)
        
        # Entry count
        entry_frame = QFrame()
        entry_frame.setStyleSheet("background-color: #1a472a; border-radius: 8px;")
        entry_layout = QVBoxLayout(entry_frame)
        entry_layout.setContentsMargins(15, 15, 15, 15)
        
        entry_label = QLabel("ENTRY")
        entry_label.setStyleSheet("color: white; font-size: 14px; background: transparent;")
        entry_label.setAlignment(Qt.AlignCenter)
        entry_layout.addWidget(entry_label)
        
        self._entry_display = QLabel("0")
        self._entry_display.setStyleSheet("color: #00ff00; font-size: 36px; font-weight: bold; background: transparent;")
        self._entry_display.setAlignment(Qt.AlignCenter)
        entry_layout.addWidget(self._entry_display)
        stats_layout.addWidget(entry_frame)
        
        stats_layout.addSpacing(10)
        
        # Exit count
        exit_frame = QFrame()
        exit_frame.setStyleSheet("background-color: #4a1a1a; border-radius: 8px;")
        exit_layout = QVBoxLayout(exit_frame)
        exit_layout.setContentsMargins(15, 15, 15, 15)
        
        exit_label = QLabel("EXIT")
        exit_label.setStyleSheet("color: white; font-size: 14px; background: transparent;")
        exit_label.setAlignment(Qt.AlignCenter)
        exit_layout.addWidget(exit_label)
        
        self._exit_display = QLabel("0")
        self._exit_display.setStyleSheet("color: #ff4444; font-size: 36px; font-weight: bold; background: transparent;")
        self._exit_display.setAlignment(Qt.AlignCenter)
        exit_layout.addWidget(self._exit_display)
        stats_layout.addWidget(exit_frame)
        
        stats_layout.addSpacing(10)
        
        # Inside count
        inside_label = QLabel("INSIDE")
        inside_label.setStyleSheet("color: #aaaaaa; font-size: 12px; background: transparent;")
        inside_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(inside_label)
        
        self._inside_display = QLabel("0")
        self._inside_display.setStyleSheet("color: white; font-size: 24px; font-weight: bold; background: transparent;")
        self._inside_display.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self._inside_display)
        
        stats_layout.addStretch()
        
        # FPS and resolution
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #3d3d3d; border-radius: 4px;")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)
        
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet("color: #aaaaaa; font-size: 11px; background: transparent;")
        self._fps_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self._fps_label)
        
        self._res_label = QLabel("Res: --")
        self._res_label.setStyleSheet("color: #aaaaaa; font-size: 11px; background: transparent;")
        self._res_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self._res_label)
        stats_layout.addWidget(info_frame)
        
        stats_layout.addSpacing(10)
        
        self._reset_btn = QPushButton("↻ Reset Counts")
        self._reset_btn.clicked.connect(self._reset_counts)
        stats_layout.addWidget(self._reset_btn)
        
        main_layout.addWidget(stats_panel)
        
        self.add_content_layout(main_layout)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self._start_btn = QPushButton("▶ Start")
        self._start_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px 16px;")
        self._start_btn.clicked.connect(self._start_counting)
        controls_layout.addWidget(self._start_btn)
        
        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.clicked.connect(self._toggle_pause)
        self._pause_btn.setEnabled(False)
        controls_layout.addWidget(self._pause_btn)
        
        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.clicked.connect(self._stop_counting)
        self._stop_btn.setEnabled(False)
        controls_layout.addWidget(self._stop_btn)
        
        controls_layout.addStretch()
        
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #666666;")
        controls_layout.addWidget(self._status_label)
        
        self._content_layout.addLayout(controls_layout)
        
        # Navigation
        self.add_back_button()
        self.add_button_stretch()
    
    def add_content_layout(self, layout):
        """Add a layout to content area."""
        self._content_layout.addLayout(layout)
    
    def set_configuration(self, source_config: SourceConfig, model_config: ModelConfig, roi_config: ROIConfiguration):
        """Set all configurations."""
        print(f"[Counting] set_configuration called")
        print(f"[Counting] Source: {source_config.source_type}, path={source_config.path}, device_id={source_config.device_id}")
        print(f"[Counting] Model: {model_config.model_name}")
        print(f"[Counting] ROI module: {roi_config.module_type}")
        
        self._source_config = source_config
        self._model_config = model_config
        self._roi_config = roi_config
        
        # Reset everything
        self._reset_counts()
        self._polyline_segments = []
        self._line_p1 = None
        self._line_p2 = None
        self._polygon_vertices = []
        
        # Reset UI
        self._video_label.setText("Click Start to begin counting...")
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Ready")
    
    def _setup_geometry(self, frame_width: int, frame_height: int):
        """Setup line/polygon geometry from ROI config."""
        print(f"[Counting] Setting up geometry for {frame_width}x{frame_height}")
        self._frame_size = (frame_width, frame_height)
        
        module_type = self._roi_config.module_type
        print(f"[Counting] Module type: {module_type}")
        
        # Reset geometry
        self._polyline_segments = []
        self._line_p1 = None
        self._line_p2 = None
        self._line_length = 0
        self._polygon_vertices = []
        
        # Handle polyline module
        if module_type == 'polyline' and self._roi_config.module_specific:
            polyline_points = self._roi_config.module_specific.get('polyline_points', [])
            print(f"[Counting] Polyline points from config: {len(polyline_points)} points")
            
            if polyline_points and len(polyline_points) >= 2:
                pixel_points = []
                for norm_pt in polyline_points:
                    px = int(norm_pt[0] * frame_width)
                    py = int(norm_pt[1] * frame_height)
                    pixel_points.append((px, py))
                
                for i in range(len(pixel_points) - 1):
                    self._polyline_segments.append((pixel_points[i], pixel_points[i + 1]))
                
                if self._roi_config.lines:
                    self._entry_side = self._roi_config.lines[0].entry_direction
                
                print(f"[Counting] Polyline setup: {len(self._polyline_segments)} segments, entry_side={self._entry_side}")
                return
        
        # Handle single line (tripwire)
        if self._roi_config.lines:
            line = self._roi_config.lines[0]
            
            self._line_p1 = (int(line.p1[0] * frame_width), int(line.p1[1] * frame_height))
            self._line_p2 = (int(line.p2[0] * frame_width), int(line.p2[1] * frame_height))
            
            dx = self._line_p2[0] - self._line_p1[0]
            dy = self._line_p2[1] - self._line_p1[1]
            self._line_length = math.sqrt(dx * dx + dy * dy)
            
            self._entry_side = line.entry_direction
            
            print(f"[Counting] Line setup: {self._line_p1} -> {self._line_p2}, length={self._line_length:.1f}, entry_side={self._entry_side}")
        
        # Handle polygons
        if self._roi_config.polygons:
            polygon = self._roi_config.polygons[0]
            self._polygon_vertices = []
            for v in polygon.vertices:
                px = int(v[0] * frame_width)
                py = int(v[1] * frame_height)
                self._polygon_vertices.append((px, py))
            print(f"[Counting] Polygon setup: {len(self._polygon_vertices)} vertices")
    
    def _start_counting(self):
        """Start the counting process."""
        print("[Counting] Start button clicked")
        
        if self._is_running:
            print("[Counting] Already running, ignoring start")
            return
        
        self._status_label.setText("Loading model...")
        self._start_btn.setEnabled(False)
        QApplication.processEvents()
        
        # Load YOLO model
        try:
            from ultralytics import YOLO
            
            model_name = self._model_config.model_name
            print(f"[Counting] Loading model: {model_name}")
            
            self._model = YOLO(model_name)
            print(f"[Counting] Model loaded successfully: {model_name}")
        except Exception as e:
            import traceback
            print(f"[Counting] Error loading model: {e}")
            traceback.print_exc()
            self._status_label.setText(f"Error loading model: {e}")
            self._start_btn.setEnabled(True)
            return
        
        # Open video source
        self._status_label.setText("Connecting to video...")
        QApplication.processEvents()
        
        print(f"[Counting] Opening video source: type={self._source_config.source_type}")
        
        try:
            if self._source_config.source_type == "rtsp":
                print(f"[Counting] RTSP URL: {self._source_config.path}")
                from app.threads.rtsp_capture_thread import RTSPCaptureThread
                self._rtsp_thread = RTSPCaptureThread(
                    rtsp_url=self._source_config.path,
                    max_width=1280,
                    max_height=720
                )
                if not self._rtsp_thread.start():
                    self._status_label.setText(f"Failed to connect: {self._rtsp_thread.error_message}")
                    self._start_btn.setEnabled(True)
                    return
                
                print("[Counting] Waiting for first RTSP frame...")
                for attempt in range(50):
                    result = self._rtsp_thread.get_frame()
                    if result:
                        frame, _ = result
                        self._setup_geometry(frame.shape[1], frame.shape[0])
                        print(f"[Counting] Got first RTSP frame: {frame.shape}")
                        break
                    time.sleep(0.1)
                else:
                    self._status_label.setText("Timeout waiting for RTSP frame")
                    self._rtsp_thread.stop()
                    self._rtsp_thread = None
                    self._start_btn.setEnabled(True)
                    return
            else:
                if self._source_config.source_type == "video_file":
                    print(f"[Counting] Opening video file: {self._source_config.path}")
                    self._cap = cv2.VideoCapture(self._source_config.path)
                else:
                    print(f"[Counting] Opening webcam: device_id={self._source_config.device_id}")
                    self._cap = cv2.VideoCapture(self._source_config.device_id)
                
                print(f"[Counting] VideoCapture created, isOpened={self._cap.isOpened() if self._cap else False}")
                
                if not self._cap or not self._cap.isOpened():
                    self._status_label.setText("Failed to open video source")
                    self._start_btn.setEnabled(True)
                    return
                
                width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[Counting] Video dimensions: {width}x{height}")
                
                if width == 0 or height == 0:
                    print("[Counting] Dimensions are 0, trying to read a frame...")
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"[Counting] Got dimensions from frame: {width}x{height}")
                    else:
                        self._status_label.setText("Cannot read from video source")
                        self._cap.release()
                        self._cap = None
                        self._start_btn.setEnabled(True)
                        return
                
                self._setup_geometry(width, height)
        
        except Exception as e:
            import traceback
            print(f"[Counting] Error opening video: {e}")
            traceback.print_exc()
            self._status_label.setText(f"Error: {e}")
            self._start_btn.setEnabled(True)
            return
        
        # Update UI
        self._is_running = True
        self._is_paused = False
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._status_label.setText("Counting in progress...")
        
        # Reset FPS counter
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        
        print("[Counting] Starting timer...")
        self._timer.start(30)
        print("[Counting] Timer started!")
    
    def _stop_counting(self):
        """Stop the counting process."""
        print("[Counting] Stopping...")
        self._timer.stop()
        self._is_running = False
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        if self._rtsp_thread:
            self._rtsp_thread.stop()
            self._rtsp_thread = None
        
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Stopped")
        print("[Counting] Stopped")
    
    def stop(self):
        """Public method to stop counting (called from MainWindow)."""
        self._stop_counting()
    
    def _toggle_pause(self):
        """Toggle pause state."""
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._pause_btn.setText("▶ Resume")
            self._status_label.setText("Paused")
        else:
            self._pause_btn.setText("⏸ Pause")
            self._status_label.setText("Counting in progress...")
    
    def _reset_counts(self):
        """Reset all counts."""
        self._entry_count = 0
        self._exit_count = 0
        self._track_states.clear()
        self._track_segment_states.clear()
        self._update_displays()
        print("[Counting] Counts reset")
    
    def _update_displays(self):
        """Update count displays."""
        self._entry_display.setText(str(self._entry_count))
        self._exit_display.setText(str(self._exit_count))
        inside = max(0, self._entry_count - self._exit_count)
        self._inside_display.setText(str(inside))
    
    def _process_frame(self):
        """Process a single frame."""
        if self._is_paused:
            return
        
        frame = None
        
        try:
            if self._source_config.source_type == "rtsp":
                if self._rtsp_thread:
                    result = self._rtsp_thread.get_frame()
                    if result:
                        frame, _ = result
            else:
                if self._cap and self._cap.isOpened():
                    ret, frame = self._cap.read()
                    if not ret:
                        if self._source_config.source_type == "video_file":
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        return
        except Exception as e:
            print(f"[Counting] Error reading frame: {e}")
            return
        
        if frame is None:
            return
        
        self._frame_count += 1
        self._fps_frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = current_time
        
        # Run YOLO detection with tracking
        try:
            results = self._model.track(
                frame,
                persist=True,
                classes=[0],
                conf=self._model_config.confidence,
                verbose=False,
                tracker="botsort.yaml"
            )
        except Exception as e:
            print(f"[Counting] Error in detection: {e}")
            return
        
        # Process detections
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate center bottom (feet position)
                    cx = (x1 + x2) // 2
                    cy = y2
                    
                    # Check for line crossing based on module type
                    if self._polyline_segments:
                        self._check_polyline_crossing(track_id, cx, cy)
                    elif self._line_p1 and self._line_p2:
                        self._check_line_crossing(track_id, cx, cy)
                    
                    # Draw bounding box
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
        
        # Draw ROI
        self._draw_roi(frame)
        
        # Draw counts on frame
        self._draw_counts(frame)
        
        # Update display
        self._display_frame(frame)
        
        # Update info
        self._fps_label.setText(f"FPS: {self._fps:.1f}")
        self._res_label.setText(f"Res: {frame.shape[1]}×{frame.shape[0]}")
    
    def _check_line_crossing(self, track_id: int, cx: int, cy: int):
        """
        Check if a track crossed the line (for tripwire module).
        Uses COOLDOWN system to allow multiple crossings by same person.
        """
        if self._line_length == 0:
            return
        
        # Calculate which side of line
        cross = self._cross_product(self._line_p1, self._line_p2, (cx, cy))
        distance = abs(cross) / self._line_length
        
        # Hysteresis zone - ignore if too close to line
        if distance < 15:
            return
        
        current_side = "side_a" if cross > 0 else "side_b"
        
        # Initialize new track
        if track_id not in self._track_states:
            self._track_states[track_id] = {
                'side': current_side,
                'cooldown': 0
            }
            return
        
        state = self._track_states[track_id]
        
        # Decrease cooldown each frame
        if state['cooldown'] > 0:
            state['cooldown'] -= 1
            return
        
        prev_side = state['side']
        
        # Check for crossing (side changed)
        if prev_side != current_side:
            if prev_side == self._entry_side:
                # Crossed from entry side = ENTRY
                self._entry_count += 1
                state['cooldown'] = self._cooldown_frames
                self._update_displays()
                print(f"[Counting] ENTRY: track {track_id}, total={self._entry_count}")
            else:
                # Crossed from exit side = EXIT
                self._exit_count += 1
                state['cooldown'] = self._cooldown_frames
                self._update_displays()
                print(f"[Counting] EXIT: track {track_id}, total={self._exit_count}")
            
            # Update side
            state['side'] = current_side
    
    def _check_polyline_crossing(self, track_id: int, cx: int, cy: int):
        """
        Check if a track crossed any polyline segment.
        Uses GLOBAL cooldown per track (not per segment) to prevent double-counting at vertices.
        """
        # Initialize track state dict for segments
        if track_id not in self._track_segment_states:
            self._track_segment_states[track_id] = {
                'global_cooldown': 0,  # Global cooldown for this track
                'segments': {}  # Per-segment side tracking
            }
        
        track_state = self._track_segment_states[track_id]
        
        # Check global cooldown first - if active, skip ALL segment checks
        if track_state['global_cooldown'] > 0:
            track_state['global_cooldown'] -= 1
            return
        
        segments_data = track_state['segments']
        
        # Check each segment
        for seg_idx, (p1, p2) in enumerate(self._polyline_segments):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            seg_length = math.sqrt(dx * dx + dy * dy)
            
            if seg_length == 0:
                continue
            
            cross = self._cross_product(p1, p2, (cx, cy))
            distance = abs(cross) / seg_length
            
            # Hysteresis - ignore if too close to line
            if distance < 15:
                continue
            
            current_side = "side_a" if cross > 0 else "side_b"
            
            # Initialize segment state
            if seg_idx not in segments_data:
                segments_data[seg_idx] = current_side
                continue
            
            prev_side = segments_data[seg_idx]
            
            # Check for crossing (side changed)
            if prev_side != current_side:
                if prev_side == self._entry_side:
                    self._entry_count += 1
                    track_state['global_cooldown'] = self._cooldown_frames  # Global cooldown
                    self._update_displays()
                    print(f"[Counting] POLYLINE ENTRY: track {track_id} on segment {seg_idx}, total={self._entry_count}")
                else:
                    self._exit_count += 1
                    track_state['global_cooldown'] = self._cooldown_frames  # Global cooldown
                    self._update_displays()
                    print(f"[Counting] POLYLINE EXIT: track {track_id} on segment {seg_idx}, total={self._exit_count}")
                
                # Update side for this segment
                segments_data[seg_idx] = current_side
                
                # IMPORTANT: Return immediately after counting to prevent double-count
                returns
    
    def _cross_product(self, p1: tuple, p2: tuple, point: tuple) -> float:
        """Calculate cross product for point relative to line."""
        return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
    
    def _draw_roi(self, frame: np.ndarray):
        """Draw ROI on frame."""
        # Draw polyline segments
        if self._polyline_segments:
            for i, (p1, p2) in enumerate(self._polyline_segments):
                cv2.line(frame, p1, p2, (0, 255, 255), 3)
                self._draw_direction_arrow(frame, p1, p2, (0, 255, 255))
            
            # Draw vertex points
            all_points = [self._polyline_segments[0][0]]
            for seg in self._polyline_segments:
                all_points.append(seg[1])
            
            for i, pt in enumerate(all_points):
                cv2.circle(frame, pt, 6, (0, 255, 255), -1)
                cv2.circle(frame, pt, 6, (255, 255, 255), 2)
        
        # Draw single line
        elif self._line_p1 and self._line_p2:
            cv2.line(frame, self._line_p1, self._line_p2, (0, 255, 255), 3)
            self._draw_direction_arrow(frame, self._line_p1, self._line_p2, (0, 255, 255))
        
        # Draw polygon
        if self._polygon_vertices and len(self._polygon_vertices) >= 3:
            pts = np.array(self._polygon_vertices, np.int32)
            cv2.polylines(frame, [pts], True, (255, 200, 0), 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (255, 200, 0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    def _draw_direction_arrow(self, frame: np.ndarray, p1: tuple, p2: tuple, color: tuple):
        """Draw direction arrow perpendicular to line."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        
        if length == 0:
            return
        
        perp_x = -dy / length
        perp_y = dx / length
        
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        offset = 25
        if self._entry_side == "side_a":
            start = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
            end = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))
        else:
            start = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))
            end = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
        
        cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.4)
    
    def _draw_counts(self, frame: np.ndarray):
        """Draw entry/exit labels on frame."""
        # Get line for label positioning
        if self._polyline_segments:
            p1, p2 = self._polyline_segments[0]
        elif self._line_p1 and self._line_p2:
            p1, p2 = self._line_p1, self._line_p2
        else:
            return
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        
        if length == 0:
            return
        
        perp_x = -dy / length
        perp_y = dx / length
        
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        offset = 60
        
        if self._entry_side == "side_a":
            entry_pos = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
            exit_pos = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))
        else:
            entry_pos = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))
            exit_pos = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
        
        cv2.putText(frame, "ENTRY", entry_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "EXIT", exit_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _display_frame(self, frame: np.ndarray):
        """Display frame on video label."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self._video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._video_label.setPixmap(scaled)
    
    def showEvent(self, event):
        """Handle page shown."""
        super().showEvent(event)
    
    def hideEvent(self, event):
        """Handle page hidden."""
        super().hideEvent(event)
        self._stop_counting()