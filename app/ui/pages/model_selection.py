"""
Model Selection Page - Choose YOLO model and processing device.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QRadioButton, QButtonGroup, QComboBox, QFrame,
    QSlider, QSpinBox
)
from PySide6.QtCore import Signal, Qt

from app.ui.pages.page_base import PageBase
from app.model.data_structures import ModelConfig
from app.model.detector import get_available_devices


# Model definitions
MODELS = {
    "cpu": [
        {"size": "n", "name": "YOLOv11n (Nano)", "desc": "Fastest, lower accuracy", "speed": "★★★★★"},
        {"size": "s", "name": "YOLOv11s (Small)", "desc": "Balanced for CPU", "speed": "★★★★☆"},
    ],
    "gpu": [
        {"size": "n", "name": "YOLOv11n (Nano)", "desc": "Very fast", "speed": "★★★★★"},
        {"size": "s", "name": "YOLOv11s (Small)", "desc": "Fast", "speed": "★★★★☆"},
        {"size": "m", "name": "YOLOv11m (Medium)", "desc": "Balanced", "speed": "★★★☆☆"},
        {"size": "l", "name": "YOLOv11l (Large)", "desc": "Accurate", "speed": "★★☆☆☆"},
        {"size": "x", "name": "YOLOv11x (XLarge)", "desc": "Most accurate", "speed": "★☆☆☆☆"},
    ]
}


class ModelSelectionPage(PageBase):
    """Page for selecting YOLO model and device."""
    
    model_configured = Signal(object)  # Emits ModelConfig
    
    def __init__(self, parent=None):
        super().__init__("Select Processing Model", parent)
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # === Device Selection ===
        device_frame = QFrame()
        device_frame.setFrameStyle(QFrame.StyledPanel)
        device_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 8px; padding: 15px; }")
        device_layout = QVBoxLayout(device_frame)
        
        device_label = QLabel("Processing Device:")
        device_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        device_layout.addWidget(device_label)
        
        self._device_group = QButtonGroup(self)
        
        # CPU option
        cpu_layout = QHBoxLayout()
        self._cpu_radio = QRadioButton("CPU")
        self._cpu_radio.setStyleSheet("font-size: 12px;")
        self._device_group.addButton(self._cpu_radio, 0)
        cpu_layout.addWidget(self._cpu_radio)
        cpu_desc = QLabel("(Works on all computers)")
        cpu_desc.setStyleSheet("color: #666;")
        cpu_layout.addWidget(cpu_desc)
        cpu_layout.addStretch()
        device_layout.addLayout(cpu_layout)
        
        # GPU option
        gpu_layout = QHBoxLayout()
        self._gpu_radio = QRadioButton("GPU (CUDA)")
        self._gpu_radio.setStyleSheet("font-size: 12px;")
        self._device_group.addButton(self._gpu_radio, 1)
        gpu_layout.addWidget(self._gpu_radio)
        
        self._gpu_combo = QComboBox()
        self._gpu_combo.setMinimumWidth(250)
        self._gpu_combo.setEnabled(False)
        gpu_layout.addWidget(self._gpu_combo)
        gpu_layout.addStretch()
        device_layout.addLayout(gpu_layout)
        
        # Populate GPU list
        devices = get_available_devices()
        gpu_available = False
        for dev in devices:
            if dev['type'] == 'cuda':
                self._gpu_combo.addItem(dev['name'], dev['id'])
                gpu_available = True
        
        if not gpu_available:
            self._gpu_radio.setEnabled(False)
            self._gpu_combo.addItem("No GPU detected")
            self._gpu_combo.setEnabled(False)
        
        self.add_content(device_frame)
        
        # === Model Selection ===
        model_frame = QFrame()
        model_frame.setFrameStyle(QFrame.StyledPanel)
        model_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 8px; padding: 15px; }")
        model_layout = QVBoxLayout(model_frame)
        
        model_label = QLabel("Model Size:")
        model_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        model_layout.addWidget(model_label)
        
        self._model_group = QButtonGroup(self)
        self._model_radios = {}
        
        # Will be populated based on device selection
        self._model_container = QVBoxLayout()
        model_layout.addLayout(self._model_container)
        
        self.add_content(model_frame)
        
        # === Confidence Threshold ===
        conf_frame = QFrame()
        conf_frame.setFrameStyle(QFrame.StyledPanel)
        conf_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 8px; padding: 15px; }")
        conf_layout = QVBoxLayout(conf_frame)
        
        conf_label = QLabel("Detection Confidence:")
        conf_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        conf_layout.addWidget(conf_label)
        
        conf_input_layout = QHBoxLayout()
        
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setMinimum(10)
        self._conf_slider.setMaximum(90)
        self._conf_slider.setValue(50)
        self._conf_slider.setTickPosition(QSlider.TicksBelow)
        self._conf_slider.setTickInterval(10)
        conf_input_layout.addWidget(self._conf_slider)
        
        self._conf_spin = QSpinBox()
        self._conf_spin.setMinimum(10)
        self._conf_spin.setMaximum(90)
        self._conf_spin.setValue(50)
        self._conf_spin.setSuffix("%")
        conf_input_layout.addWidget(self._conf_spin)
        
        conf_layout.addLayout(conf_input_layout)
        
        conf_hint = QLabel("Lower = more detections (may include false positives). Higher = fewer but more confident detections.")
        conf_hint.setStyleSheet("color: #666; font-size: 11px;")
        conf_hint.setWordWrap(True)
        conf_layout.addWidget(conf_hint)
        
        # Connect slider and spinbox
        self._conf_slider.valueChanged.connect(self._conf_spin.setValue)
        self._conf_spin.valueChanged.connect(self._conf_slider.setValue)
        
        self.add_content(conf_frame)
        
        self.add_stretch()
        
        # Connect device selection
        self._device_group.buttonClicked.connect(self._on_device_changed)
        
        # Default to CPU
        self._cpu_radio.setChecked(True)
        self._on_device_changed(self._cpu_radio)
        
        # Buttons
        self.add_back_button()
        self.add_button_stretch()
        self._next_btn = self.add_next_button()
        self._next_btn.clicked.connect(self._on_next)
    
    def _on_device_changed(self, button):
        """Handle device selection change."""
        is_gpu = self._device_group.id(button) == 1
        self._gpu_combo.setEnabled(is_gpu)
        
        # Clear model options
        for i in reversed(range(self._model_container.count())):
            item = self._model_container.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        
        # Clear button group
        for btn in self._model_group.buttons():
            self._model_group.removeButton(btn)
        
        # Add models for selected device
        models = MODELS["gpu"] if is_gpu else MODELS["cpu"]
        
        for i, model in enumerate(models):
            model_layout = QHBoxLayout()
            
            radio = QRadioButton(model["name"])
            self._model_group.addButton(radio, i)
            model_layout.addWidget(radio)
            
            speed_label = QLabel(f"Speed: {model['speed']}")
            speed_label.setStyleSheet("color: #666; margin-left: 20px;")
            model_layout.addWidget(speed_label)
            
            desc_label = QLabel(f"({model['desc']})")
            desc_label.setStyleSheet("color: #999;")
            model_layout.addWidget(desc_label)
            
            model_layout.addStretch()
            
            container = QWidget()
            container.setLayout(model_layout)
            self._model_container.addWidget(container)
            
            self._model_radios[model["size"]] = radio
        
        # Select first model by default
        if models:
            first_radio = self._model_group.button(0)
            if first_radio:
                first_radio.setChecked(True)
    
    def _on_next(self):
        """Handle next button."""
        is_gpu = self._device_group.checkedId() == 1
        
        # Get selected model size
        models = MODELS["gpu"] if is_gpu else MODELS["cpu"]
        model_idx = self._model_group.checkedId()
        model_size = models[model_idx]["size"] if model_idx >= 0 else "n"
        
        # Get GPU device ID
        gpu_id = self._gpu_combo.currentData() if is_gpu else 0
        
        config = ModelConfig(
            mode="gpu" if is_gpu else "cpu",
            device_id=gpu_id if gpu_id is not None else 0,
            model_size=model_size,
            confidence=self._conf_spin.value() / 100.0
        )
        
        self.model_configured.emit(config)