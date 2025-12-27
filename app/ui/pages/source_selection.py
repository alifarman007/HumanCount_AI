"""
Source Selection Page - Choose video source (file, RTSP, webcam).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QRadioButton, QButtonGroup, QLineEdit, QPushButton,
    QFileDialog, QComboBox, QFrame, QMessageBox
)
from PySide6.QtCore import Signal
import cv2

from app.ui.pages.page_base import PageBase
from app.model.data_structures import SourceConfig
from app.model.video_source import enumerate_webcams


class SourceSelectionPage(PageBase):
    """Page for selecting video source."""
    
    source_configured = Signal(object)  # Emits SourceConfig
    
    def __init__(self, parent=None):
        super().__init__("Select Video Source", parent)
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Radio button group
        self._source_group = QButtonGroup(self)
        
        # === Video File Option ===
        file_frame = QFrame()
        file_frame.setFrameStyle(QFrame.StyledPanel)
        file_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }")
        file_layout = QVBoxLayout(file_frame)
        
        self._file_radio = QRadioButton("Video File")
        self._file_radio.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._source_group.addButton(self._file_radio, 0)
        file_layout.addWidget(self._file_radio)
        
        file_input_layout = QHBoxLayout()
        self._file_path = QLineEdit()
        self._file_path.setPlaceholderText("Select a video file...")
        self._file_path.setEnabled(False)
        file_input_layout.addWidget(self._file_path)
        
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.setEnabled(False)
        self._browse_btn.clicked.connect(self._browse_file)
        file_input_layout.addWidget(self._browse_btn)
        file_layout.addLayout(file_input_layout)
        
        self.add_content(file_frame)
        
        # === RTSP Stream Option ===
        rtsp_frame = QFrame()
        rtsp_frame.setFrameStyle(QFrame.StyledPanel)
        rtsp_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }")
        rtsp_layout = QVBoxLayout(rtsp_frame)
        
        self._rtsp_radio = QRadioButton("RTSP Stream")
        self._rtsp_radio.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._source_group.addButton(self._rtsp_radio, 1)
        rtsp_layout.addWidget(self._rtsp_radio)
        
        self._rtsp_url = QLineEdit()
        self._rtsp_url.setPlaceholderText("rtsp://username:password@192.168.1.100:554/stream1")
        self._rtsp_url.setEnabled(False)
        rtsp_layout.addWidget(self._rtsp_url)
        
        rtsp_hint = QLabel("Example: rtsp://admin:admin@192.168.1.64:554/Streaming/Channels/101")
        rtsp_hint.setStyleSheet("color: #666; font-size: 11px;")
        rtsp_layout.addWidget(rtsp_hint)
        
        self.add_content(rtsp_frame)
        
        # === Webcam Option ===
        webcam_frame = QFrame()
        webcam_frame.setFrameStyle(QFrame.StyledPanel)
        webcam_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }")
        webcam_layout = QVBoxLayout(webcam_frame)
        
        webcam_header = QHBoxLayout()
        self._webcam_radio = QRadioButton("Webcam")
        self._webcam_radio.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._source_group.addButton(self._webcam_radio, 2)
        webcam_header.addWidget(self._webcam_radio)
        
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setMaximumWidth(80)
        self._refresh_btn.clicked.connect(self._refresh_webcams)
        webcam_header.addWidget(self._refresh_btn)
        webcam_header.addStretch()
        webcam_layout.addLayout(webcam_header)
        
        self._webcam_combo = QComboBox()
        self._webcam_combo.setEnabled(False)
        webcam_layout.addWidget(self._webcam_combo)
        
        self.add_content(webcam_frame)
        
        # Connect radio buttons
        self._source_group.buttonClicked.connect(self._on_source_type_changed)
        
        # Select webcam by default and refresh list
        self._webcam_radio.setChecked(True)
        self._on_source_type_changed(self._webcam_radio)
        self._refresh_webcams()
        
        self.add_stretch()
        
        # Test connection button
        test_layout = QHBoxLayout()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._test_connection)
        test_layout.addWidget(self._test_btn)
        
        self._test_status = QLabel("")
        test_layout.addWidget(self._test_status)
        test_layout.addStretch()
        self._content_layout.addLayout(test_layout)
        
        # Buttons
        self.add_back_button()
        self.add_button_stretch()
        self._next_btn = self.add_next_button()
        self._next_btn.clicked.connect(self._on_next)
    
    def _on_source_type_changed(self, button):
        """Handle source type change."""
        source_id = self._source_group.id(button)
        
        # Enable/disable inputs based on selection
        self._file_path.setEnabled(source_id == 0)
        self._browse_btn.setEnabled(source_id == 0)
        self._rtsp_url.setEnabled(source_id == 1)
        self._webcam_combo.setEnabled(source_id == 2)
        
        # Clear test status if it exists
        if hasattr(self, '_test_status'):
            self._test_status.setText("")
    
    def _browse_file(self):
        """Open file browser."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)"
        )
        if file_path:
            self._file_path.setText(file_path)
    
    def _refresh_webcams(self):
        """Refresh webcam list."""
        self._webcam_combo.clear()
        
        webcams = enumerate_webcams()
        if webcams:
            for device_id, name in webcams:
                self._webcam_combo.addItem(name, device_id)
        else:
            self._webcam_combo.addItem("No webcams found", -1)
    
    def _get_current_source(self) -> tuple:
        """Get current source type and path/id."""
        source_id = self._source_group.checkedId()
        
        if source_id == 0:  # Video file
            return ("video_file", self._file_path.text())
        elif source_id == 1:  # RTSP
            return ("rtsp", self._rtsp_url.text())
        else:  # Webcam
            device_id = self._webcam_combo.currentData()
            return ("webcam", device_id if device_id is not None else 0)
    
    def _test_connection(self):
        """Test the selected source."""
        source_type, source = self._get_current_source()
        
        self._test_status.setText("Testing...")
        self._test_status.setStyleSheet("color: #666;")
        self._test_btn.setEnabled(False)
        
        # Force UI update
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        try:
            if source_type == "video_file":
                if not source:
                    raise ValueError("No file selected")
                cap = cv2.VideoCapture(source)
            elif source_type == "rtsp":
                if not source:
                    raise ValueError("No RTSP URL entered")
                cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    self._test_status.setText(f"✓ Connected: {width}x{height} @ {fps:.1f}fps")
                    self._test_status.setStyleSheet("color: green; font-weight: bold;")
                else:
                    self._test_status.setText("✗ Could not read frame")
                    self._test_status.setStyleSheet("color: red;")
                cap.release()
            else:
                self._test_status.setText("✗ Could not connect")
                self._test_status.setStyleSheet("color: red;")
                
        except Exception as e:
            self._test_status.setText(f"✗ Error: {str(e)}")
            self._test_status.setStyleSheet("color: red;")
        
        self._test_btn.setEnabled(True)
    
    def _on_next(self):
        """Handle next button."""
        source_type, source = self._get_current_source()
        
        # Validate
        if source_type == "video_file" and not source:
            QMessageBox.warning(self, "Error", "Please select a video file.")
            return
        elif source_type == "rtsp" and not source:
            QMessageBox.warning(self, "Error", "Please enter an RTSP URL.")
            return
        elif source_type == "webcam" and source == -1:
            QMessageBox.warning(self, "Error", "No webcam available.")
            return
        
        # Create config
        config = SourceConfig(
            source_type=source_type,
            path=source if source_type in ("video_file", "rtsp") else "",
            device_id=source if source_type == "webcam" else 0
        )
        
        self.source_configured.emit(config)