"""
HumanCount AI - People Counting Software
Main entry point for the application.
"""

import sys
import os

# Suppress OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from app.ui.main_window import MainWindow


def main():
    """Main application entry point."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("HumanCount AI")
    app.setOrganizationName("HumanCount")
    app.setOrganizationDomain("humancount.ai")
    
    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Set style
    app.setStyle("Fusion")
    
    # Apply dark theme stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QFrame {
            background-color: #ffffff;
        }
        QPushButton {
            padding: 6px 12px;
            border-radius: 4px;
            border: 1px solid #ddd;
            background-color: #ffffff;
        }
        QPushButton:hover {
            background-color: #e9ecef;
            border-color: #adb5bd;
        }
        QPushButton:pressed {
            background-color: #dee2e6;
        }
        QPushButton:disabled {
            background-color: #e9ecef;
            color: #6c757d;
        }
        QLineEdit, QComboBox {
            padding: 6px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: #ffffff;
        }
        QLineEdit:focus, QComboBox:focus {
            border-color: #0066cc;
        }
        QRadioButton {
            spacing: 8px;
        }
        QRadioButton::indicator {
            width: 16px;
            height: 16px;
        }
        QScrollArea {
            border: none;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #dee2e6;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #0066cc;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        QSlider::sub-page:horizontal {
            background: #0066cc;
            border-radius: 3px;
        }
    """)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()