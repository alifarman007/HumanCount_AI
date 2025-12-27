"""
ROI Configuration Page - Draw lines/polygons on live video with direction selection.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QComboBox, QMessageBox, QButtonGroup
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2
import numpy as np
import math

from app.ui.pages.page_base import PageBase
from app.model.data_structures import (
    SourceConfig, ModelConfig, ROIConfiguration, 
    LineConfig, DebounceConfig
)

from app.threads.rtsp_capture_thread import RTSPCaptureThread

class VideoCanvas(QLabel):
    """Widget for displaying video and drawing ROI."""
    
    line_drawn = Signal(tuple, tuple)  # p1, p2 in normalized coords
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Connecting to video source...")
        
        # Drawing state
        self._drawing = False
        self._start_point = None
        self._end_point = None
        self._current_line = None  # (p1, p2) in normalized coords
        
        # Direction: which side is entry
        # "side_a" = left/top side of line, "side_b" = right/bottom side
        self._entry_side = "side_a"
        
        # Video frame
        self._current_frame = None
        self._frame_size = (640, 480)
        
        self.setMouseTracking(True)
    
    def set_entry_side(self, side: str):
        """Set which side is the entry side."""
        self._entry_side = side
        self._update_display()
    
    def set_frame(self, frame):
        """Set current video frame."""
        self._current_frame = frame
        self._frame_size = (frame.shape[1], frame.shape[0])
        self._update_display()
    
    def _update_display(self):
        """Update the display with frame and drawings."""
        if self._current_frame is None:
            return
        
        frame = self._current_frame.copy()
        h, w = frame.shape[:2]
        
        # Draw existing line
        if self._current_line:
            p1_norm, p2_norm = self._current_line
            p1_px = (int(p1_norm[0] * w), int(p1_norm[1] * h))
            p2_px = (int(p2_norm[0] * w), int(p2_norm[1] * h))
            
            # Draw main line (yellow)
            cv2.line(frame, p1_px, p2_px, (0, 255, 255), 3)
            
            # Calculate perpendicular direction for labels
            dx = p2_px[0] - p1_px[0]
            dy = p2_px[1] - p1_px[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Perpendicular unit vector (rotated 90 degrees)
                perp_x = -dy / length
                perp_y = dx / length
                
                # Midpoint of line
                mid_x = (p1_px[0] + p2_px[0]) // 2
                mid_y = (p1_px[1] + p2_px[1]) // 2
                
                # Label positions (offset from line)
                offset = 60
                side_a_x = int(mid_x + perp_x * offset)
                side_a_y = int(mid_y + perp_y * offset)
                side_b_x = int(mid_x - perp_x * offset)
                side_b_y = int(mid_y - perp_y * offset)
                
                # Determine entry/exit labels based on selection
                if self._entry_side == "side_a":
                    entry_pos = (side_a_x, side_a_y)
                    exit_pos = (side_b_x, side_b_y)
                else:
                    entry_pos = (side_b_x, side_b_y)
                    exit_pos = (side_a_x, side_a_y)
                
                # Draw ENTRY label (green)
                cv2.putText(frame, "ENTRY", (entry_pos[0] - 40, entry_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Draw EXIT label (red)
                cv2.putText(frame, "EXIT", (exit_pos[0] - 30, exit_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw arrow from entry side to exit side
                arrow_start_x = int(mid_x + perp_x * 25 * (1 if self._entry_side == "side_a" else -1))
                arrow_start_y = int(mid_y + perp_y * 25 * (1 if self._entry_side == "side_a" else -1))
                arrow_end_x = int(mid_x - perp_x * 25 * (1 if self._entry_side == "side_a" else -1))
                arrow_end_y = int(mid_y - perp_y * 25 * (1 if self._entry_side == "side_a" else -1))
                
                cv2.arrowedLine(frame, (arrow_start_x, arrow_start_y), 
                               (arrow_end_x, arrow_end_y), (0, 255, 0), 3, tipLength=0.4)
        
        # Draw line being drawn (preview)
        if self._drawing and self._start_point and self._end_point:
            cv2.line(frame, self._start_point, self._end_point, (255, 255, 0), 2)
        
        # Convert to QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
    
    def _widget_to_frame_coords(self, pos) -> tuple:
        """Convert widget coordinates to frame coordinates."""
        if self._current_frame is None:
            return (0, 0)
        
        pixmap = self.pixmap()
        if pixmap is None:
            return (0, 0)
        
        # Calculate offset (pixmap is centered)
        x_offset = (self.width() - pixmap.width()) // 2
        y_offset = (self.height() - pixmap.height()) // 2
        
        # Adjust for offset
        x = pos.x() - x_offset
        y = pos.y() - y_offset
        
        # Scale to frame coordinates
        scale_x = self._frame_size[0] / pixmap.width()
        scale_y = self._frame_size[1] / pixmap.height()
        
        frame_x = int(x * scale_x)
        frame_y = int(y * scale_y)
        
        # Clamp to frame bounds
        frame_x = max(0, min(frame_x, self._frame_size[0] - 1))
        frame_y = max(0, min(frame_y, self._frame_size[1] - 1))
        
        return (frame_x, frame_y)
    
    def mousePressEvent(self, event):
        """Start drawing line."""
        if event.button() == Qt.LeftButton and self._current_frame is not None:
            self._drawing = True
            self._start_point = self._widget_to_frame_coords(event.pos())
            self._end_point = self._start_point
    
    def mouseMoveEvent(self, event):
        """Update line endpoint while drawing."""
        if self._drawing:
            self._end_point = self._widget_to_frame_coords(event.pos())
            self._update_display()
    
    def mouseReleaseEvent(self, event):
        """Finish drawing line."""
        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            self._end_point = self._widget_to_frame_coords(event.pos())
            
            # Only create line if it has some length
            dx = abs(self._end_point[0] - self._start_point[0])
            dy = abs(self._end_point[1] - self._start_point[1])
            
            if dx > 20 or dy > 20:
                # Store as normalized coordinates
                w, h = self._frame_size
                p1_norm = (self._start_point[0] / w, self._start_point[1] / h)
                p2_norm = (self._end_point[0] / w, self._end_point[1] / h)
                self._current_line = (p1_norm, p2_norm)
                
                self.line_drawn.emit(p1_norm, p2_norm)
            
            self._update_display()
    
    def clear_drawing(self):
        """Clear current drawing."""
        self._current_line = None
        self._start_point = None
        self._end_point = None
        self._entry_side = "side_a"
        self._update_display()
    
    def get_line(self):
        """Get current line in normalized coordinates."""
        return self._current_line
    
    def get_entry_side(self):
        """Get selected entry side."""
        return self._entry_side


class ROIConfigurationPage(PageBase):
    """Page for configuring ROI by drawing on video."""
    
    roi_configured = Signal(object)  # Emits ROIConfiguration
    
    def __init__(self, parent=None):
        super().__init__("Configure Detection Zone", parent)
        
        self._source_config = None
        self._model_config = None
        self._module_type = None
        self._cap = None
        self._rtsp_thread = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_frame)
        
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Instructions
        self._instructions = QLabel(
            "Step 1: Draw a line across the path where people will cross.\n"
            "Step 2: Select which side people ENTER from."
        )
        self._instructions.setStyleSheet(
            "color: #0066cc; font-size: 12px; padding: 10px; "
            "background: #e7f1ff; border-radius: 4px;"
        )
        self._instructions.setWordWrap(True)
        self.add_content(self._instructions)
        
        # Video canvas
        self._canvas = VideoCanvas()
        self._canvas.line_drawn.connect(self._on_line_drawn)
        self.add_content(self._canvas)
        
        # Direction selection (hidden until line is drawn)
        self._direction_frame = QFrame()
        self._direction_frame.setStyleSheet(
            "QFrame { background-color: #fff3cd; border-radius: 4px; padding: 10px; }"
        )
        self._direction_frame.setVisible(False)
        
        direction_layout = QHBoxLayout(self._direction_frame)
        
        direction_label = QLabel("Entry comes from:")
        direction_label.setStyleSheet("font-weight: bold;")
        direction_layout.addWidget(direction_label)
        
        self._direction_group = QButtonGroup(self)
        
        self._side_a_btn = QPushButton("◀ This Side (Green)")
        self._side_a_btn.setCheckable(True)
        self._side_a_btn.setChecked(True)
        self._side_a_btn.setStyleSheet("""
            QPushButton { background-color: #d4edda; border: 2px solid #28a745; }
            QPushButton:checked { background-color: #28a745; color: white; }
        """)
        self._direction_group.addButton(self._side_a_btn, 0)
        direction_layout.addWidget(self._side_a_btn)
        
        self._side_b_btn = QPushButton("This Side (Green) ▶")
        self._side_b_btn.setCheckable(True)
        self._side_b_btn.setStyleSheet("""
            QPushButton { background-color: #d4edda; border: 2px solid #28a745; }
            QPushButton:checked { background-color: #28a745; color: white; }
        """)
        self._direction_group.addButton(self._side_b_btn, 1)
        direction_layout.addWidget(self._side_b_btn)
        
        self._swap_btn = QPushButton("⇄ Swap Direction")
        self._swap_btn.clicked.connect(self._swap_direction)
        direction_layout.addWidget(self._swap_btn)
        
        direction_layout.addStretch()
        
        self._direction_group.buttonClicked.connect(self._on_direction_changed)
        
        self.add_content(self._direction_frame)
        
        # Drawing controls
        controls_layout = QHBoxLayout()
        
        self._clear_btn = QPushButton("✕ Clear & Redraw")
        self._clear_btn.clicked.connect(self._clear_drawing)
        controls_layout.addWidget(self._clear_btn)
        
        controls_layout.addStretch()
        
        self._status_label = QLabel("Draw a line to continue")
        self._status_label.setStyleSheet("color: #666;")
        controls_layout.addWidget(self._status_label)
        
        self._content_layout.addLayout(controls_layout)
        
        # Buttons
        self.add_back_button()
        self.add_button_stretch()
        self._next_btn = self.add_next_button("Start Counting →")
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._on_next)
    
    def set_configuration(self, module_type: str, source_config: SourceConfig, model_config: ModelConfig):
        """Set configuration and start video preview."""
        self._module_type = module_type
        self._source_config = source_config
        self._model_config = model_config
        
        # Reset state
        self._canvas.clear_drawing()
        self._direction_frame.setVisible(False)
        self._next_btn.setEnabled(False)
        self._status_label.setText("Draw a line to continue")
        
        # Start video capture
        self._start_video()
    
    def _start_video(self):
        """Start video capture for preview."""
        # Stop any existing capture
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        if hasattr(self, '_rtsp_thread') and self._rtsp_thread is not None:
            self._rtsp_thread.stop()
            self._rtsp_thread = None
        
        # Check if RTSP
        if self._source_config.source_type == "rtsp":
            # Use threaded RTSP capture
            from app.threads.rtsp_capture_thread import RTSPCaptureThread
            
            self._rtsp_thread = RTSPCaptureThread(
                rtsp_url=self._source_config.path,
                target_width=1280,
                target_height=720
            )
            
            if self._rtsp_thread.start():
                self._timer.start(33)
            else:
                self._canvas.setText(f"Failed to connect: {self._rtsp_thread.error_message}")
        else:
            # Use direct capture for webcam/file
            if self._source_config.source_type == "video_file":
                self._cap = cv2.VideoCapture(self._source_config.path)
            else:
                self._cap = cv2.VideoCapture(self._source_config.device_id)
            
            if self._cap.isOpened():
                self._timer.start(33)
            else:
                self._canvas.setText("Failed to connect to video source")
    
    def _update_frame(self):
        """Update video frame."""
        frame = None
        
        if self._source_config.source_type == "rtsp":
            # RTSP mode
            if hasattr(self, '_rtsp_thread') and self._rtsp_thread:
                result = self._rtsp_thread.get_frame()
                if result:
                    frame, _ = result
        else:
            # Webcam/file mode
            if self._cap is None or not self._cap.isOpened():
                return
            
            ret, frame = self._cap.read()
            if not ret:
                if self._source_config.source_type == "video_file":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
        
        if frame is not None:
            self._canvas.set_frame(frame)
    
    def _on_line_drawn(self, p1, p2):
        """Handle line drawn."""
        self._direction_frame.setVisible(True)
        self._status_label.setText("✓ Line drawn - select entry direction")
        self._status_label.setStyleSheet("color: green; font-weight: bold;")
        self._next_btn.setEnabled(True)
        
        # Reset direction to default
        self._side_a_btn.setChecked(True)
        self._canvas.set_entry_side("side_a")
    
    def _on_direction_changed(self, button):
        """Handle direction selection change."""
        if button == self._side_a_btn:
            self._canvas.set_entry_side("side_a")
        else:
            self._canvas.set_entry_side("side_b")
    
    def _swap_direction(self):
        """Swap entry/exit direction."""
        if self._side_a_btn.isChecked():
            self._side_b_btn.setChecked(True)
            self._canvas.set_entry_side("side_b")
        else:
            self._side_a_btn.setChecked(True)
            self._canvas.set_entry_side("side_a")
    
    def _clear_drawing(self):
        """Clear drawing and reset."""
        self._canvas.clear_drawing()
        self._direction_frame.setVisible(False)
        self._next_btn.setEnabled(False)
        self._status_label.setText("Draw a line to continue")
        self._status_label.setStyleSheet("color: #666;")
    
    def _on_next(self):
        """Handle next button."""
        line = self._canvas.get_line()
        
        if line is None:
            QMessageBox.warning(self, "Error", "Please draw a line first.")
            return
        
        # Stop video
        self._timer.stop()
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        if hasattr(self, '_rtsp_thread') and self._rtsp_thread:
            self._rtsp_thread.stop()
            self._rtsp_thread = None
        
        # Determine entry direction based on selection
        entry_side = self._canvas.get_entry_side()
        
        # Create ROI config
        p1, p2 = line
        roi_config = ROIConfiguration(
            module_type=self._module_type,
            lines=[LineConfig(
                id="main_line",
                p1=p1,
                p2=p2,
                entry_direction=entry_side
            )],
            debounce=DebounceConfig(
                hysteresis_distance=25,
                min_frames_stable=2,
                cooldown_frames=25
            )
        )
        
        self.roi_configured.emit(roi_config)
    
    def showEvent(self, event):
        """Handle page shown."""
        super().showEvent(event)
        if self._source_config:
            self._start_video()
    
    def hideEvent(self, event):
        """Handle page hidden."""
        super().hideEvent(event)
        self._timer.stop()
        
        if hasattr(self, '_rtsp_thread') and self._rtsp_thread:
            self._rtsp_thread.stop()
            self._rtsp_thread = None