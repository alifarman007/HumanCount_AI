"""
Module Selection Page - Choose counting module type.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QRadioButton, QButtonGroup, QFrame, QScrollArea
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont

from app.ui.pages.page_base import PageBase


# Module definitions
MODULES = [
    {
        "type": "tripwire",
        "name": "Simple Tripwire",
        "description": "Count crossings over a single line. Best for doors, corridors, and turnstiles.",
        "draws": "1 Line",
        "icon": "━━━━━"
    },
    {
        "type": "polyline",
        "name": "Multi-Segment Polyline",
        "description": "Count crossings over connected line segments. Best for L-shaped entrances and corners.",
        "draws": "3+ Points",
        "icon": "━━╲╲━━"
    },
    {
        "type": "dual_gate",
        "name": "Dual-Gate Logic",
        "description": "Two separate lines with directional enforcement. Best for one-way security doors.",
        "draws": "2 Lines",
        "icon": "━━  ━━"
    },
    {
        "type": "zone_transition",
        "name": "Zone Transition",
        "description": "Two adjacent zones requiring sequential passage. Best for wide lobbies and malls.",
        "draws": "2 Polygons",
        "icon": "▢ → ▢"
    },
    {
        "type": "perimeter",
        "name": "Virtual Perimeter",
        "description": "Single zone boundary detection. Best for kiosks, counters, and display areas.",
        "draws": "1 Polygon",
        "icon": "◇◇◇◇"
    },
    {
        "type": "occupancy",
        "name": "Real-time Occupancy",
        "description": "Count people currently inside a zone (not crossings). Best for rooms and elevators.",
        "draws": "1 Polygon",
        "icon": "[5/10]"
    },
    {
        "type": "hybrid_gate_parking",
        "name": "Hybrid Gate + Parking",
        "description": "Line crossing + parking zone detection. Best for factory gates with vehicle parking.",
        "draws": "1 Line + 1 Polygon",
        "icon": "━━ + ▢"
    },
]


class ModuleCard(QFrame):
    """Card widget for a single module option."""
    
    def __init__(self, module_data: dict, parent=None):
        super().__init__(parent)
        
        self._module_data = module_data
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup card UI."""
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            ModuleCard {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
            }
            ModuleCard:hover {
                border-color: #0066cc;
                background-color: #e7f1ff;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setSpacing(15)
        
        # Radio button
        self.radio = QRadioButton()
        layout.addWidget(self.radio)
        
        # Icon
        icon_label = QLabel(self._module_data["icon"])
        icon_label.setFont(QFont("Consolas", 14))
        icon_label.setMinimumWidth(80)
        layout.addWidget(icon_label)
        
        # Text content
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        
        # Name
        name_label = QLabel(self._module_data["name"])
        name_font = QFont()
        name_font.setPointSize(11)
        name_font.setBold(True)
        name_label.setFont(name_font)
        text_layout.addWidget(name_label)
        
        # Description
        desc_label = QLabel(self._module_data["description"])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666;")
        text_layout.addWidget(desc_label)
        
        # Draws info
        draws_label = QLabel(f"You draw: {self._module_data['draws']}")
        draws_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        text_layout.addWidget(draws_label)
        
        layout.addLayout(text_layout, 1)
    
    @property
    def module_type(self) -> str:
        return self._module_data["type"]
    
    def mousePressEvent(self, event):
        """Select radio when card is clicked."""
        self.radio.setChecked(True)
        super().mousePressEvent(event)


class ModuleSelectionPage(PageBase):
    """Page for selecting counting module type."""
    
    module_selected = Signal(str)  # Emits module type
    
    def __init__(self, parent=None):
        super().__init__("Select Counting Module", parent)
        
        self._selected_type = None
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Subtitle
        subtitle = QLabel("Choose the counting method that best fits your setup:")
        subtitle.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        self.add_content(subtitle)
        
        # Scroll area for module cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)
        
        # Button group for radio buttons
        self._button_group = QButtonGroup(self)
        
        # Create module cards
        self._cards = []
        for i, module in enumerate(MODULES):
            card = ModuleCard(module)
            self._button_group.addButton(card.radio, i)
            scroll_layout.addWidget(card)
            self._cards.append(card)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        
        self.add_content(scroll)
        
        # Connect button group
        self._button_group.buttonClicked.connect(self._on_selection_changed)
        
        # Buttons
        self.add_button_stretch()
        self._next_btn = self.add_next_button()
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._on_next)
        
        # Select first by default
        self._cards[0].radio.setChecked(True)
        self._on_selection_changed(self._cards[0].radio)
    
    def _on_selection_changed(self, button):
        """Handle selection change."""
        idx = self._button_group.id(button)
        if idx >= 0:
            self._selected_type = MODULES[idx]["type"]
            self._next_btn.setEnabled(True)
    
    def _on_next(self):
        """Handle next button click."""
        if self._selected_type:
            self.module_selected.emit(self._selected_type)