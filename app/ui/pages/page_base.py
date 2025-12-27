"""
Base class for all pages.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont


class PageBase(QWidget):
    """Base class for all wizard-style pages."""
    
    # Common signals
    back_requested = Signal()
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        
        self._title = title
        self._setup_base_ui()
    
    def _setup_base_ui(self):
        """Setup base page structure."""
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(20, 20, 20, 20)
        self._main_layout.setSpacing(15)
        
        # Title
        self._title_label = QLabel(self._title)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._main_layout.addWidget(self._title_label)
        
        # Content area (subclasses add to this)
        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(0, 10, 0, 10)
        self._main_layout.addWidget(self._content_widget, 1)
        
        # Button bar at bottom
        self._button_bar = QHBoxLayout()
        self._button_bar.setSpacing(10)
        self._main_layout.addLayout(self._button_bar)
    
    def add_content(self, widget: QWidget):
        """Add widget to content area."""
        self._content_layout.addWidget(widget)
    
    def add_stretch(self):
        """Add stretch to content area."""
        self._content_layout.addStretch()
    
    def add_back_button(self) -> QPushButton:
        """Add back button to button bar."""
        btn = QPushButton("← Back")
        btn.setMinimumWidth(100)
        btn.clicked.connect(self.back_requested.emit)
        self._button_bar.addWidget(btn)
        return btn
    
    def add_button_stretch(self):
        """Add stretch to button bar."""
        self._button_bar.addStretch()
    
    def add_next_button(self, text: str = "Next →") -> QPushButton:
        """Add next/continue button to button bar."""
        btn = QPushButton(text)
        btn.setMinimumWidth(120)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self._button_bar.addWidget(btn)
        return btn