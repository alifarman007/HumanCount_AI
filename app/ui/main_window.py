"""
Main Window - Central application window with stacked pages.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QStackedWidget,
    QMessageBox
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent

from app.model.data_structures import AppConfiguration


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Contains stacked widget for page navigation:
    1. Module Selection
    2. Source Selection  
    3. Model Selection
    4. ROI Configuration
    5. Counting
    """
    
    # Page indices
    PAGE_MODULE_SELECT = 0
    PAGE_SOURCE_SELECT = 1
    PAGE_MODEL_SELECT = 2
    PAGE_ROI_CONFIG = 3
    PAGE_COUNTING = 4
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("HumanCount AI - People Counter")
        self.setMinimumSize(1024, 768)
        
        # Current configuration being built
        self._config = AppConfiguration()
        
        # Setup UI
        self._setup_ui()
        
        # Start at module selection
        self._stack.setCurrentIndex(self.PAGE_MODULE_SELECT)
    
    def _setup_ui(self):
        """Setup the main UI structure."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Stacked widget for pages
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)
        
        # Create pages (will be replaced with actual page widgets)
        self._create_placeholder_pages()
    
    def _create_placeholder_pages(self):
        """Create placeholder pages (will be replaced)."""
        from app.ui.pages.module_selection import ModuleSelectionPage
        from app.ui.pages.source_selection import SourceSelectionPage
        from app.ui.pages.model_selection import ModelSelectionPage
        from app.ui.pages.roi_configuration import ROIConfigurationPage
        from app.ui.pages.counting_page import CountingPage
        
        # Page 0: Module Selection
        self._module_page = ModuleSelectionPage()
        self._module_page.module_selected.connect(self._on_module_selected)
        self._stack.addWidget(self._module_page)
        
        # Page 1: Source Selection
        self._source_page = SourceSelectionPage()
        self._source_page.source_configured.connect(self._on_source_configured)
        self._source_page.back_requested.connect(lambda: self._go_to_page(self.PAGE_MODULE_SELECT))
        self._stack.addWidget(self._source_page)
        
        # Page 2: Model Selection
        self._model_page = ModelSelectionPage()
        self._model_page.model_configured.connect(self._on_model_configured)
        self._model_page.back_requested.connect(lambda: self._go_to_page(self.PAGE_SOURCE_SELECT))
        self._stack.addWidget(self._model_page)
        
        # Page 3: ROI Configuration
        self._roi_page = ROIConfigurationPage()
        self._roi_page.roi_configured.connect(self._on_roi_configured)
        self._roi_page.back_requested.connect(lambda: self._go_to_page(self.PAGE_MODEL_SELECT))
        self._stack.addWidget(self._roi_page)
        
        # Page 4: Counting
        self._counting_page = CountingPage()
        self._counting_page.back_requested.connect(self._on_back_from_counting)
        self._stack.addWidget(self._counting_page)
    
    def _go_to_page(self, page_index: int):
        """Navigate to a specific page."""
        self._stack.setCurrentIndex(page_index)
    
    @Slot(str)
    def _on_module_selected(self, module_type: str):
        """Handle module selection."""
        print(f"[MainWindow] Module selected: {module_type}")  # DEBUG
        self._config.module_type = module_type
        self._go_to_page(self.PAGE_SOURCE_SELECT)
    
    @Slot(object)
    def _on_source_configured(self, source_config):
        """Handle source configuration."""
        self._config.source = source_config
        self._go_to_page(self.PAGE_MODEL_SELECT)
    
    @Slot(object)
    def _on_model_configured(self, model_config):
        """Handle model configuration."""
        print(f"[MainWindow] Model configured, module_type is: {self._config.module_type}")  # DEBUG
        self._config.model = model_config
        
        # Pass config to ROI page
        self._roi_page.set_configuration(
            module_type=self._config.module_type,
            source_config=self._config.source,
            model_config=self._config.model
        )
        
        self._go_to_page(self.PAGE_ROI_CONFIG)
    
    @Slot(object)
    def _on_roi_configured(self, roi_config):
        """Handle ROI configuration."""
        print(f"[MainWindow] ROI configured received!")  # DEBUG
        self._config.roi = roi_config
        
        # Set configuration on counting page
        self._counting_page.set_configuration(
            source_config=self._config.source,
            model_config=self._config.model,
            roi_config=self._config.roi
        )
        
        self._go_to_page(self.PAGE_COUNTING)
        print(f"[MainWindow] Navigated to counting page")  # DEBUG
    
    @Slot()
    def _on_back_from_counting(self):
        """Handle back from counting page."""
        # Stop any running processes
        self._counting_page.stop()
        self._go_to_page(self.PAGE_ROI_CONFIG)
    
    def closeEvent(self, event: QCloseEvent):
        """Handle window close."""
        # Stop counting if running
        if hasattr(self, '_counting_page'):
            self._counting_page.stop()
        
        event.accept()