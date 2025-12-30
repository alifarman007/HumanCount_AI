"""
ROI Configuration Page - Draw lines/polygons on live video with direction selection.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QMessageBox, QSpinBox
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QFont
import cv2

from app.ui.pages.page_base import PageBase
from app.ui.widgets.drawing_canvas import DrawingCanvas, DrawingMode
from app.model.data_structures import (
    SourceConfig, ModelConfig, ROIConfiguration, 
    LineConfig, PolygonConfig, DebounceConfig
)


# Module to drawing mode mapping
MODULE_DRAWING_MODES = {
    'tripwire': DrawingMode.LINE,
    'polyline': DrawingMode.POLYLINE,
    'dual_gate': DrawingMode.DUAL_LINE,
    'zone_transition': DrawingMode.DUAL_POLYGON,
    'perimeter': DrawingMode.POLYGON,
    'occupancy': DrawingMode.POLYGON,
    'hybrid_gate_parking': DrawingMode.LINE_AND_POLYGON,
}

# Instructions per module
MODULE_INSTRUCTIONS = {
    'tripwire': "Draw a line across the path. Click and drag to draw.",
    'polyline': "Click to add points. Double-click or click 'Finish' button to complete.",
    'dual_gate': "Draw ENTRY gate first (green), then EXIT gate (red). Click and drag.",
    'zone_transition': "Draw ZONE A first, then ZONE B. Click to add points, then Finish.",
    'perimeter': "Click to add polygon points. Double-click or click 'Finish' to close.",
    'occupancy': "Click to add polygon points. Double-click or click 'Finish' to close.",
    'hybrid_gate_parking': "Step 1: Draw gate line (drag). Step 2: Draw parking zone polygon.",
}


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
        
        self._ui_ready = False
        self._setup_content()
        self._ui_ready = True
    
    def _setup_content(self):
        """Setup page content."""
        # Instructions
        self._instructions = QLabel("Draw on the video to configure detection zone.")
        self._instructions.setStyleSheet(
            "color: #0066cc; font-size: 12px; padding: 10px; "
            "background: #e7f1ff; border-radius: 4px;"
        )
        self._instructions.setWordWrap(True)
        self.add_content(self._instructions)
        
        # Drawing canvas
        self._canvas = DrawingCanvas()
        self.add_content(self._canvas)
        
        # Direction controls
        self._direction_frame = QFrame()
        self._direction_frame.setStyleSheet(
            "QFrame { background-color: #fff3cd; border-radius: 4px; padding: 10px; }"
        )
        self._direction_frame.setVisible(False)
        
        direction_layout = QHBoxLayout(self._direction_frame)
        direction_label = QLabel("Entry direction:")
        direction_label.setStyleSheet("font-weight: bold;")
        direction_layout.addWidget(direction_label)
        
        self._swap_btn = QPushButton("⇄ Swap Entry/Exit Sides")
        self._swap_btn.clicked.connect(self._swap_direction)
        direction_layout.addWidget(self._swap_btn)
        direction_layout.addStretch()
        
        self.add_content(self._direction_frame)
        
        # Capacity settings (for occupancy module)
        self._capacity_frame = QFrame()
        self._capacity_frame.setStyleSheet(
            "QFrame { background-color: #d4edda; border-radius: 4px; padding: 10px; }"
        )
        self._capacity_frame.setVisible(False)
        
        capacity_layout = QHBoxLayout(self._capacity_frame)
        capacity_label = QLabel("Max Capacity:")
        capacity_label.setStyleSheet("font-weight: bold;")
        capacity_layout.addWidget(capacity_label)
        
        self._capacity_spin = QSpinBox()
        self._capacity_spin.setRange(1, 9999)
        self._capacity_spin.setValue(50)
        capacity_layout.addWidget(self._capacity_spin)
        capacity_layout.addStretch()
        
        self.add_content(self._capacity_frame)
        
        # Drawing controls
        controls_layout = QHBoxLayout()
        
        self._clear_btn = QPushButton("✕ Clear All")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        controls_layout.addWidget(self._clear_btn)
        
        self._undo_btn = QPushButton("↩ Undo Point")
        self._undo_btn.clicked.connect(self._on_undo_clicked)
        self._undo_btn.setVisible(False)
        controls_layout.addWidget(self._undo_btn)
        
        self._finish_btn = QPushButton("✓ Finish Shape")
        self._finish_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold; padding: 8px 16px; }"
            "QPushButton:disabled { background-color: #6c757d; }"
        )
        self._finish_btn.clicked.connect(self._on_finish_clicked)
        self._finish_btn.setVisible(False)
        self._finish_btn.setEnabled(False)
        controls_layout.addWidget(self._finish_btn)
        
        self._next_shape_btn = QPushButton("Next Shape →")
        self._next_shape_btn.setStyleSheet(
            "QPushButton { background-color: #007bff; color: white; font-weight: bold; padding: 8px 16px; }"
        )
        self._next_shape_btn.setVisible(False)
        self._next_shape_btn.clicked.connect(self._advance_to_next_shape)
        controls_layout.addWidget(self._next_shape_btn)
        
        controls_layout.addStretch()
        
        self._status_label = QLabel("Draw to continue")
        self._status_label.setStyleSheet("color: #666;")
        controls_layout.addWidget(self._status_label)
        
        self._content_layout.addLayout(controls_layout)
        
        # Navigation buttons
        self.add_back_button()
        self.add_button_stretch()
        self._next_btn = self.add_next_button("Start Counting →")
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._on_next)
        
        # Connect canvas signals AFTER all widgets created
        self._canvas.drawing_updated.connect(self._on_drawing_updated)
        self._canvas.drawing_complete.connect(self._on_drawing_complete)
    
    def _on_clear_clicked(self):
        """Handle clear button click."""
        self._canvas.clear_all()
    
    def _on_undo_clicked(self):
        """Handle undo button click."""
        self._canvas.undo_last_point()
    
    def _on_finish_clicked(self):
        """Handle finish button click."""
        print("[ROI Page] Finish button clicked")
        self._canvas.finish_current_shape()
    
    def set_configuration(self, module_type: str, source_config: SourceConfig, model_config: ModelConfig):
        """Set configuration and start video preview."""
        print(f"[ROI Page] set_configuration called with module_type: {module_type}")
        
        self._module_type = module_type
        self._source_config = source_config
        self._model_config = model_config
        
        # Set drawing mode
        drawing_mode = MODULE_DRAWING_MODES.get(module_type, DrawingMode.LINE)
        print(f"[ROI Page] Drawing mode will be: {drawing_mode}")
        
        self._canvas.set_mode(drawing_mode)
        
        # Update instructions
        instructions = MODULE_INSTRUCTIONS.get(module_type, "Draw on the video.")
        print(f"[ROI Page] Instructions: {instructions}")
        self._instructions.setText(instructions)
        
        # Show/hide controls
        self._direction_frame.setVisible(module_type in ['tripwire', 'polyline', 'dual_gate', 'hybrid_gate_parking'])
        self._capacity_frame.setVisible(module_type == 'occupancy')
        
        # Show undo/finish for point-based modes
        is_point_mode = module_type in ['polyline', 'perimeter', 'occupancy', 'zone_transition']
        self._undo_btn.setVisible(is_point_mode)
        self._finish_btn.setVisible(is_point_mode)
        self._finish_btn.setEnabled(False)
        
        # For hybrid, buttons will be shown/hidden dynamically
        if module_type == 'hybrid_gate_parking':
            self._undo_btn.setVisible(False)
            self._finish_btn.setVisible(False)
        
        # Reset state
        self._next_btn.setEnabled(False)
        self._next_shape_btn.setVisible(False)
        self._status_label.setText("Draw to continue")
        self._status_label.setStyleSheet("color: #666;")
        
        # Start video
        self._start_video()
    
    def _start_video(self):
        """Start video capture."""
        self._stop_video()
        
        if self._source_config.source_type == "rtsp":
            from app.threads.rtsp_capture_thread import RTSPCaptureThread
            
            self._rtsp_thread = RTSPCaptureThread(
                rtsp_url=self._source_config.path,
                max_width=1280,
                max_height=720
            )
            
            if self._rtsp_thread.start():
                self._timer.start(33)
            else:
                self._canvas.setText(f"Failed to connect: {self._rtsp_thread.error_message}")
        else:
            if self._source_config.source_type == "video_file":
                self._cap = cv2.VideoCapture(self._source_config.path)
            else:
                self._cap = cv2.VideoCapture(self._source_config.device_id)
            
            if self._cap.isOpened():
                self._timer.start(33)
            else:
                self._canvas.setText("Failed to connect to video source")
    
    def _stop_video(self):
        """Stop video capture."""
        self._timer.stop()
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        if self._rtsp_thread:
            self._rtsp_thread.stop()
            self._rtsp_thread = None
    
    def _update_frame(self):
        """Update video frame."""
        frame = None
        
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
        
        if frame is not None:
            self._canvas.set_frame(frame)
    
    def _on_drawing_updated(self):
        """Handle drawing updates."""
        if not self._ui_ready:
            return
        
        data = self._canvas.get_drawing_data()
        mode = self._canvas.get_mode()
        is_complete = data.get('complete', False)
        
        # Reset status style
        self._status_label.setStyleSheet("color: #666;")
        
        if mode == DrawingMode.POLYLINE:
            n = len(data.get('polyline_points', []))
            is_finished = data.get('finished', False)
            
            if is_complete:
                self._finish_btn.setVisible(False)
                self._status_label.setText(f"✓ Polyline complete ({n} points)")
                self._status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self._finish_btn.setVisible(True)
                self._finish_btn.setEnabled(n >= 2)
                self._finish_btn.setText(f"✓ Finish Polyline ({n} pts)")
                self._status_label.setText(f"Click to add points ({n} added)")
        
        elif mode == DrawingMode.POLYGON:
            n = len(data.get('polygon1', []))
            
            if is_complete:
                self._finish_btn.setVisible(False)
                self._status_label.setText(f"✓ Polygon complete ({n} points)")
                self._status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self._finish_btn.setVisible(True)
                self._finish_btn.setEnabled(n >= 3)
                self._finish_btn.setText(f"✓ Close Polygon ({n} pts)")
                self._status_label.setText(f"Click to add points ({n} added, need 3+)")
        
        elif mode == DrawingMode.DUAL_POLYGON:
            idx = self._canvas.get_current_shape_index()
            n1 = len(data.get('polygon1', []))
            n2 = len(data.get('polygon2', []))
            p1_done = data.get('polygon1_finished', False)
            
            if is_complete:
                self._finish_btn.setVisible(False)
                self._status_label.setText("✓ Both zones complete!")
                self._status_label.setStyleSheet("color: green; font-weight: bold;")
            elif idx == 0:
                self._finish_btn.setVisible(True)
                self._finish_btn.setEnabled(n1 >= 3)
                self._finish_btn.setText(f"✓ Finish Zone A ({n1} pts)")
                self._status_label.setText(f"Drawing Zone A ({n1} points)")
            else:
                self._finish_btn.setVisible(True)
                self._finish_btn.setEnabled(n2 >= 3)
                self._finish_btn.setText(f"✓ Finish Zone B ({n2} pts)")
                self._status_label.setText(f"Drawing Zone B ({n2} points)")
        
        elif mode == DrawingMode.LINE_AND_POLYGON:
            idx = self._canvas.get_current_shape_index()
            n = len(data.get('polygon1', []))
            
            if is_complete:
                self._finish_btn.setVisible(False)
                self._next_shape_btn.setVisible(False)
                self._undo_btn.setVisible(False)
                self._status_label.setText("✓ Gate and parking zone complete!")
                self._status_label.setStyleSheet("color: green; font-weight: bold;")
            elif idx == 0:
                if data.get('line1'):
                    self._next_shape_btn.setVisible(True)
                    self._next_shape_btn.setText("Next: Draw Parking Zone →")
                    self._finish_btn.setVisible(False)
                    self._undo_btn.setVisible(False)
                    self._status_label.setText("Line done! Click 'Next' to draw parking zone.")
                else:
                    self._next_shape_btn.setVisible(False)
                    self._finish_btn.setVisible(False)
                    self._undo_btn.setVisible(False)
                    self._status_label.setText("Draw the gate line (click and drag)")
            else:
                self._next_shape_btn.setVisible(False)
                self._finish_btn.setVisible(True)
                self._finish_btn.setEnabled(n >= 3)
                self._finish_btn.setText(f"✓ Finish Zone ({n} pts)")
                self._undo_btn.setVisible(True)
                self._status_label.setText(f"Drawing parking zone ({n} points)")
        
        elif mode == DrawingMode.DUAL_LINE:
            if is_complete:
                self._status_label.setText("✓ Both gates complete!")
                self._status_label.setStyleSheet("color: green; font-weight: bold;")
            elif data.get('line1') and not data.get('line2'):
                self._status_label.setText("Entry gate done. Now draw EXIT gate (red).")
            else:
                self._status_label.setText("Draw ENTRY gate first (green).")
        
        elif mode == DrawingMode.LINE:
            if is_complete:
                self._status_label.setText("✓ Line complete!")
                self._status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self._status_label.setText("Click and drag to draw a line.")
        
        # Update next button
        self._next_btn.setEnabled(is_complete)
    
    def _on_drawing_complete(self, data: dict):
        """Handle drawing completion."""
        print("[ROI Page] Drawing complete signal received")
        self._status_label.setText("✓ Drawing complete!")
        self._status_label.setStyleSheet("color: green; font-weight: bold;")
        self._next_btn.setEnabled(True)
        self._finish_btn.setVisible(False)
        self._next_shape_btn.setVisible(False)
    
    def _swap_direction(self):
        """Swap entry/exit direction."""
        current = self._canvas._line1_entry_side
        new_side = "side_b" if current == "side_a" else "side_a"
        self._canvas.set_entry_side(new_side, 0)
    
    def _advance_to_next_shape(self):
        """Advance to next shape."""
        self._canvas.advance_to_next_shape()
        self._next_shape_btn.setVisible(False)
        self._finish_btn.setVisible(True)
        self._finish_btn.setEnabled(False)
        self._undo_btn.setVisible(True)
        self._status_label.setText("Now draw the parking zone polygon.")
    
    def _on_next(self):
        """Handle next button."""
        print("[ROI Page] Next button clicked")  # DEBUG
        
        data = self._canvas.get_drawing_data()
        print(f"[ROI Page] Drawing data: complete={data.get('complete')}")  # DEBUG
        
        if not data.get('complete'):
            QMessageBox.warning(self, "Error", "Please complete the drawing first.")
            return
        
        # Stop video
        self._stop_video()
        
        # Build ROI configuration
        roi_config = self._build_roi_config(data)
        print(f"[ROI Page] ROI config built, emitting signal...")  # DEBUG
        
        self.roi_configured.emit(roi_config)
        print(f"[ROI Page] Signal emitted!")  # DEBUG
    
    def _build_roi_config(self, data: dict) -> ROIConfiguration:
        """Build ROI configuration from drawing data."""
        lines = []
        polygons = []
        module_specific = {}
        
        if data.get('line1'):
            p1, p2 = data['line1']
            lines.append(LineConfig(
                id="line_1",
                p1=p1,
                p2=p2,
                entry_direction=data.get('line1_entry_side', 'side_a'),
                label="Entry Gate" if self._module_type == 'dual_gate' else ""
            ))
        
        if data.get('line2'):
            p1, p2 = data['line2']
            lines.append(LineConfig(
                id="line_2",
                p1=p1,
                p2=p2,
                entry_direction=data.get('line2_entry_side', 'side_a'),
                label="Exit Gate"
            ))
        
        if data.get('polyline_points'):
            points = data['polyline_points']
            lines.append(LineConfig(
                id="polyline",
                p1=points[0],
                p2=points[-1],
                entry_direction=data.get('entry_side', 'side_a')
            ))
            module_specific['polyline_points'] = points
        
        if data.get('polygon1') and len(data['polygon1']) >= 3:
            polygons.append(PolygonConfig(
                id="polygon_1",
                vertices=data['polygon1'],
                label="Zone A" if self._module_type == 'zone_transition' else "Detection Zone",
                zone_type="detection"
            ))
        
        if data.get('polygon2') and len(data['polygon2']) >= 3:
            polygons.append(PolygonConfig(
                id="polygon_2",
                vertices=data['polygon2'],
                label="Zone B",
                zone_type="confirmation"
            ))
        
        if self._module_type == 'occupancy':
            module_specific['max_capacity'] = self._capacity_spin.value()
            module_specific['warning_threshold'] = 0.8
            module_specific['critical_threshold'] = 0.96
        
        if self._module_type == 'hybrid_gate_parking':
            module_specific['vanish_timeout_seconds'] = 3.0
            module_specific['fps'] = 30
        
        return ROIConfiguration(
            module_type=self._module_type,
            lines=lines,
            polygons=polygons,
            debounce=DebounceConfig(
                hysteresis_distance=25,
                min_frames_stable=2,
                cooldown_frames=25,
                min_frames_in_zone=3
            ),
            module_specific=module_specific
        )
    
    def showEvent(self, event):
        """Handle page shown."""
        super().showEvent(event)
        if self._source_config:
            self._start_video()
    
    def hideEvent(self, event):
        """Handle page hidden."""
        super().hideEvent(event)
        self._stop_video()