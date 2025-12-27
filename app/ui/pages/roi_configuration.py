"""
ROI Configuration Page - Draw lines/polygons on live video.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QComboBox, QMessageBox
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2
import numpy as np

from app.ui.pages.page_base import PageBase
from app.model.data_structures import (
    SourceConfig, ModelConfig, ROIConfiguration, 
    LineConfig, DebounceConfig
)


class VideoCanvas(QLabel):
    """Widget for displaying video and drawing ROI."""
    
    line_drawn = Signal(tuple, tuple)  # p1, p2
    
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
        
        # Video frame
        self._current_frame = None
        self._frame_size = (640, 480)
        
        self.setMouseTracking(True)
    
    def set_frame(self, frame):
        """Set current video frame."""
        self._current_frame = frame
        self._frame_size = (frame.shape[1], frame.shape[0])
        self._update_display()
    
    def _update_display(self):
        """Update the display with frame and drawings."""
        if self._current_frame is None:
            return
        
        # Convert frame to QImage
        frame = self._current_frame.copy()
        
        # Draw existing line
        if self._current_line:
            p1, p2 = self._current_line
            p1_px = (int(p1[0] * frame.shape[1]), int(p1[1] * frame.shape[0]))
            p2_px = (int(p2[0] * frame.shape[1]), int(p2[1] * frame.shape[0]))
            cv2.line(frame, p1_px, p2_px, (0, 255, 255), 3)
            
            # Draw direction arrow
            mid_x = (p1_px[0] + p2_px[0]) // 2
            mid_y = (p1_px[1] + p2_px[1]) // 2
            cv2.arrowedLine(frame, (mid_x - 40, mid_y), (mid_x + 40, mid_y), 
                           (0, 255, 0), 2, tipLength=0.3)
            
            # Labels
            cv2.putText(frame, "ENTRY", (50, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "EXIT", (frame.shape[1] - 100, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw line being drawn
        if self._drawing and self._start_point and self._end_point:
            cv2.line(frame, self._start_point, self._end_point, (255, 255, 0), 2)
        
        # Convert to QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
    
    def _widget_to_frame_coords(self, pos) -> tuple:
        """Convert widget coordinates to frame coordinates."""
        if self._current_frame is None:
            return (0, 0)
        
        # Get the pixmap rect (centered in label)
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
        self._update_display()
    
    def get_line(self):
        """Get current line in normalized coordinates."""
        return self._current_line


class ROIConfigurationPage(PageBase):
    """Page for configuring ROI by drawing on video."""
    
    roi_configured = Signal(object)  # Emits ROIConfiguration
    
    def __init__(self, parent=None):
        super().__init__("Configure Detection Zone", parent)
        
        self._source_config = None
        self._model_config = None
        self._module_type = None
        self._cap = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_frame)
        
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Instructions
        self._instructions = QLabel(
            "Draw a vertical line on the video. People crossing LEFT→RIGHT = Entry, RIGHT→LEFT = Exit"
        )
        self._instructions.setStyleSheet("color: #0066cc; font-size: 12px; padding: 5px; background: #e7f1ff; border-radius: 4px;")
        self._instructions.setWordWrap(True)
        self.add_content(self._instructions)
        
        # Video canvas
        self._canvas = VideoCanvas()
        self._canvas.line_drawn.connect(self._on_line_drawn)
        self.add_content(self._canvas)
        
        # Drawing controls
        controls_layout = QHBoxLayout()
        
        self._clear_btn = QPushButton("Clear Drawing")
        self._clear_btn.clicked.connect(self._canvas.clear_drawing)
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
        
        # Start video capture
        self._start_video()
    
    def _start_video(self):
        """Start video capture for preview."""
        if self._cap is not None:
            self._cap.release()
        
        # Open video source
        if self._source_config.source_type == "video_file":
            self._cap = cv2.VideoCapture(self._source_config.path)
        elif self._source_config.source_type == "rtsp":
            self._cap = cv2.VideoCapture(self._source_config.path)
        else:
            self._cap = cv2.VideoCapture(self._source_config.device_id)
        
        if self._cap.isOpened():
            self._timer.start(33)  # ~30 fps
        else:
            self._canvas.setText("Failed to connect to video source")
    
    def _update_frame(self):
        """Update video frame."""
        if self._cap is None or not self._cap.isOpened():
            return
        
        ret, frame = self._cap.read()
        if ret:
            self._canvas.set_frame(frame)
        else:
            # For video files, loop back
            if self._source_config.source_type == "video_file":
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def _on_line_drawn(self, p1, p2):
        """Handle line drawn."""
        self._status_label.setText("✓ Line configured")
        self._status_label.setStyleSheet("color: green; font-weight: bold;")
        self._next_btn.setEnabled(True)
    
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
        
        # Create ROI config
        p1, p2 = line
        roi_config = ROIConfiguration(
            module_type=self._module_type,
            lines=[LineConfig(
                id="main_line",
                p1=p1,
                p2=p2,
                entry_direction="left_to_right"
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