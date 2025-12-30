"""
Drawing Canvas Widget - Supports lines, polylines, and polygons.
"""

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal, Qt, QPoint, QTimer
from PySide6.QtGui import QImage, QPixmap, QMouseEvent
import cv2
import numpy as np
import math
from typing import List, Optional, Tuple
from enum import Enum, auto


class DrawingMode(Enum):
    """Drawing modes for different module types."""
    NONE = auto()
    LINE = auto()           # Single line (Module 1)
    POLYLINE = auto()       # Connected line segments (Module 2)
    DUAL_LINE = auto()      # Two separate lines (Module 3)
    POLYGON = auto()        # Single polygon (Module 5, 6)
    DUAL_POLYGON = auto()   # Two polygons (Module 4)
    LINE_AND_POLYGON = auto()  # Line + polygon (Module 7)


class DrawingCanvas(QLabel):
    """
    Canvas for drawing ROI shapes on video.
    """
    
    # Signals
    drawing_complete = Signal(dict)
    drawing_updated = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Connecting to video source...")
        
        # Drawing mode
        self._mode = DrawingMode.NONE
        
        # Current drawing state
        self._is_drawing = False
        self._temp_point = None
        self._drag_start = None
        
        # Stored shapes
        self._line1: Optional[Tuple[tuple, tuple]] = None
        self._line2: Optional[Tuple[tuple, tuple]] = None
        self._polyline_points: List[tuple] = []
        self._polygon1_points: List[tuple] = []
        self._polygon2_points: List[tuple] = []
        
        # Which shape is being drawn (for dual modes)
        self._current_shape_index = 0
        
        # Track if shape is explicitly finished by user
        self._polyline_finished = False
        self._polygon1_finished = False
        self._polygon2_finished = False
        
        # Direction settings
        self._line1_entry_side = "side_a"
        self._line2_entry_side = "side_a"
        
        # Video frame
        self._current_frame: Optional[np.ndarray] = None
        self._frame_size = (640, 480)
        
        # Double-click detection
        self._last_click_time = 0
        self._double_click_threshold = 300  # ms
        self._pending_point = None
        self._click_timer = QTimer()
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._process_pending_point)
        
        self.setMouseTracking(True)
    
    def set_mode(self, mode: DrawingMode):
        """Set the drawing mode."""
        print(f"[DrawingCanvas] Setting mode to: {mode}")
        self._mode = mode
        self.clear_all()
    
    def get_mode(self) -> DrawingMode:
        """Get current drawing mode."""
        return self._mode
    
    def set_frame(self, frame: np.ndarray):
        """Set current video frame."""
        self._current_frame = frame
        self._frame_size = (frame.shape[1], frame.shape[0])
        self._update_display()
    
    def set_entry_side(self, side: str, line_index: int = 0):
        """Set entry side for a line."""
        if line_index == 0:
            self._line1_entry_side = side
        else:
            self._line2_entry_side = side
        self._update_display()
    
    def clear_all(self):
        """Clear all drawings."""
        self._line1 = None
        self._line2 = None
        self._polyline_points = []
        self._polygon1_points = []
        self._polygon2_points = []
        self._current_shape_index = 0
        self._is_drawing = False
        self._temp_point = None
        self._drag_start = None
        
        # Reset finished flags
        self._polyline_finished = False
        self._polygon1_finished = False
        self._polygon2_finished = False
        
        # Cancel any pending click
        self._click_timer.stop()
        self._pending_point = None
        
        self._update_display()
        self.drawing_updated.emit()
    
    def advance_to_next_shape(self):
        """Move to next shape (for multi-shape modes)."""
        self._current_shape_index = 1
        self._is_drawing = False
        self._update_display()
        self.drawing_updated.emit()
    
    def get_current_shape_index(self) -> int:
        """Get current shape index."""
        return self._current_shape_index
    
    def _widget_to_normalized(self, pos: QPoint) -> tuple:
        """Convert widget coordinates to normalized (0-1) coordinates."""
        if self._current_frame is None:
            return (0, 0)
        
        pixmap = self.pixmap()
        if pixmap is None:
            return (0, 0)
        
        x_offset = (self.width() - pixmap.width()) // 2
        y_offset = (self.height() - pixmap.height()) // 2
        
        x = pos.x() - x_offset
        y = pos.y() - y_offset
        
        x = max(0, min(x, pixmap.width() - 1))
        y = max(0, min(y, pixmap.height() - 1))
        
        norm_x = x / pixmap.width() if pixmap.width() > 0 else 0
        norm_y = y / pixmap.height() if pixmap.height() > 0 else 0
        
        return (norm_x, norm_y)
    
    def _normalized_to_pixels(self, norm_point: tuple, frame_shape: tuple) -> tuple:
        """Convert normalized coords to pixel coords."""
        h, w = frame_shape[:2]
        return (int(norm_point[0] * w), int(norm_point[1] * h))
    
    def _process_pending_point(self):
        """Process a pending point after double-click timeout."""
        if self._pending_point is not None:
            self._add_point(self._pending_point)
            self._pending_point = None
    
    def _add_point(self, pos: tuple):
        """Add a point to the current shape."""
        if self._mode == DrawingMode.POLYLINE:
            if not self._polyline_finished:
                self._polyline_points.append(pos)
                self._is_drawing = True
        
        elif self._mode == DrawingMode.POLYGON:
            if not self._polygon1_finished:
                self._polygon1_points.append(pos)
                self._is_drawing = True
        
        elif self._mode == DrawingMode.DUAL_POLYGON:
            if self._current_shape_index == 0:
                if not self._polygon1_finished:
                    self._polygon1_points.append(pos)
                    self._is_drawing = True
            else:
                if not self._polygon2_finished:
                    self._polygon2_points.append(pos)
                    self._is_drawing = True
        
        elif self._mode == DrawingMode.LINE_AND_POLYGON:
            if self._current_shape_index == 1:
                if not self._polygon1_finished:
                    self._polygon1_points.append(pos)
                    self._is_drawing = True
        
        self._update_display()
        self.drawing_updated.emit()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() != Qt.LeftButton or self._current_frame is None:
            return
        
        pos = self._widget_to_normalized(event.pos())
        
        # For line-based drawing (drag mode)
        if self._mode == DrawingMode.LINE:
            self._is_drawing = True
            self._drag_start = pos
            self._line1 = (pos, pos)
            return
        
        elif self._mode == DrawingMode.DUAL_LINE:
            self._is_drawing = True
            self._drag_start = pos
            if self._current_shape_index == 0:
                self._line1 = (pos, pos)
            else:
                self._line2 = (pos, pos)
            return
        
        elif self._mode == DrawingMode.LINE_AND_POLYGON:
            if self._current_shape_index == 0:
                # Drawing line (drag mode)
                self._is_drawing = True
                self._drag_start = pos
                self._line1 = (pos, pos)
                return
        
        # For point-based drawing, use timer to detect double-click
        if self._mode in (DrawingMode.POLYLINE, DrawingMode.POLYGON, 
                          DrawingMode.DUAL_POLYGON, DrawingMode.LINE_AND_POLYGON):
            # Cancel any pending point and timer
            self._click_timer.stop()
            self._pending_point = pos
            # Start timer - if no double-click within threshold, add the point
            self._click_timer.start(self._double_click_threshold)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if self._current_frame is None:
            return
        
        self._temp_point = self._widget_to_normalized(event.pos())
        self._update_display()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() != Qt.LeftButton:
            return
        
        pos = self._widget_to_normalized(event.pos())
        
        # Handle line drawing modes (drag to draw)
        if self._mode == DrawingMode.LINE:
            if self._drag_start:
                dx = abs(pos[0] - self._drag_start[0])
                dy = abs(pos[1] - self._drag_start[1])
                if dx > 0.03 or dy > 0.03:
                    self._line1 = (self._drag_start, pos)
                    self._check_drawing_complete()
                else:
                    self._line1 = None
            self._is_drawing = False
            self._drag_start = None
            self._update_display()
            self.drawing_updated.emit()
        
        elif self._mode == DrawingMode.DUAL_LINE:
            if self._drag_start:
                dx = abs(pos[0] - self._drag_start[0])
                dy = abs(pos[1] - self._drag_start[1])
                if dx > 0.03 or dy > 0.03:
                    if self._current_shape_index == 0:
                        self._line1 = (self._drag_start, pos)
                        self._current_shape_index = 1
                    else:
                        self._line2 = (self._drag_start, pos)
                        self._check_drawing_complete()
                else:
                    if self._current_shape_index == 0:
                        self._line1 = None
                    else:
                        self._line2 = None
            self._is_drawing = False
            self._drag_start = None
            self._update_display()
            self.drawing_updated.emit()
        
        elif self._mode == DrawingMode.LINE_AND_POLYGON:
            if self._current_shape_index == 0 and self._drag_start:
                dx = abs(pos[0] - self._drag_start[0])
                dy = abs(pos[1] - self._drag_start[1])
                if dx > 0.03 or dy > 0.03:
                    self._line1 = (self._drag_start, pos)
                else:
                    self._line1 = None
                self._is_drawing = False
                self._drag_start = None
                self._update_display()
                self.drawing_updated.emit()
    
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double click (finish polyline/polygon)."""
        if event.button() != Qt.LeftButton:
            return
        
        print(f"[DrawingCanvas] Double click detected, mode={self._mode}")
        
        # Cancel the pending single-click point
        self._click_timer.stop()
        self._pending_point = None
        
        # Finish the current shape
        self._finish_shape_internal()
    
    def _finish_shape_internal(self):
        """Internal method to finish the current shape."""
        print(f"[DrawingCanvas] Finishing shape, mode={self._mode}")
        
        if self._mode == DrawingMode.POLYLINE:
            if len(self._polyline_points) >= 2:
                self._polyline_finished = True
                self._is_drawing = False
                print(f"[DrawingCanvas] Polyline finished with {len(self._polyline_points)} points")
                self._check_drawing_complete()
        
        elif self._mode == DrawingMode.POLYGON:
            if len(self._polygon1_points) >= 3:
                self._polygon1_finished = True
                self._is_drawing = False
                print(f"[DrawingCanvas] Polygon finished with {len(self._polygon1_points)} points")
                self._check_drawing_complete()
        
        elif self._mode == DrawingMode.DUAL_POLYGON:
            if self._current_shape_index == 0:
                if len(self._polygon1_points) >= 3:
                    self._polygon1_finished = True
                    self._current_shape_index = 1
                    self._is_drawing = False
                    print(f"[DrawingCanvas] Zone A finished, moving to Zone B")
            else:
                if len(self._polygon2_points) >= 3:
                    self._polygon2_finished = True
                    self._is_drawing = False
                    print(f"[DrawingCanvas] Zone B finished")
                    self._check_drawing_complete()
        
        elif self._mode == DrawingMode.LINE_AND_POLYGON:
            if self._current_shape_index == 1:
                if len(self._polygon1_points) >= 3:
                    self._polygon1_finished = True
                    self._is_drawing = False
                    print(f"[DrawingCanvas] Parking zone finished")
                    self._check_drawing_complete()
        
        self._update_display()
        self.drawing_updated.emit()
    
    def finish_current_shape(self):
        """Public method to finish the current shape (called by external button)."""
        print(f"[DrawingCanvas] Finish button clicked, mode={self._mode}")
        self._finish_shape_internal()
    
    def undo_last_point(self):
        """Remove the last added point."""
        if self._mode == DrawingMode.POLYLINE:
            if self._polyline_points and not self._polyline_finished:
                self._polyline_points.pop()
        
        elif self._mode == DrawingMode.POLYGON:
            if self._polygon1_points and not self._polygon1_finished:
                self._polygon1_points.pop()
        
        elif self._mode == DrawingMode.DUAL_POLYGON:
            if self._current_shape_index == 0:
                if self._polygon1_points and not self._polygon1_finished:
                    self._polygon1_points.pop()
            else:
                if self._polygon2_points and not self._polygon2_finished:
                    self._polygon2_points.pop()
        
        elif self._mode == DrawingMode.LINE_AND_POLYGON:
            if self._current_shape_index == 1:
                if self._polygon1_points and not self._polygon1_finished:
                    self._polygon1_points.pop()
        
        self._update_display()
        self.drawing_updated.emit()
    
    def _check_drawing_complete(self):
        """Check if drawing is complete and emit signal."""
        data = self.get_drawing_data()
        print(f"[DrawingCanvas] Checking complete: {data.get('complete')}")
        if data.get('complete', False):
            self.drawing_complete.emit(data)
    
    def get_drawing_data(self) -> dict:
        """Get all drawing data."""
        data = {
            'mode': self._mode.name,
            'complete': False
        }
        
        if self._mode == DrawingMode.LINE:
            if self._line1:
                data['line1'] = self._line1
                data['line1_entry_side'] = self._line1_entry_side
                data['complete'] = True
        
        elif self._mode == DrawingMode.POLYLINE:
            data['polyline_points'] = self._polyline_points.copy()
            data['entry_side'] = self._line1_entry_side
            data['finished'] = self._polyline_finished
            if len(self._polyline_points) >= 2 and self._polyline_finished:
                data['complete'] = True
        
        elif self._mode == DrawingMode.DUAL_LINE:
            data['line1'] = self._line1
            data['line1_entry_side'] = self._line1_entry_side
            data['line2'] = self._line2
            data['line2_entry_side'] = self._line2_entry_side
            if self._line1 and self._line2:
                data['complete'] = True
        
        elif self._mode == DrawingMode.POLYGON:
            data['polygon1'] = self._polygon1_points.copy()
            data['finished'] = self._polygon1_finished
            if len(self._polygon1_points) >= 3 and self._polygon1_finished:
                data['complete'] = True
        
        elif self._mode == DrawingMode.DUAL_POLYGON:
            data['polygon1'] = self._polygon1_points.copy()
            data['polygon2'] = self._polygon2_points.copy()
            data['polygon1_finished'] = self._polygon1_finished
            data['polygon2_finished'] = self._polygon2_finished
            if (len(self._polygon1_points) >= 3 and self._polygon1_finished and
                len(self._polygon2_points) >= 3 and self._polygon2_finished):
                data['complete'] = True
        
        elif self._mode == DrawingMode.LINE_AND_POLYGON:
            data['line1'] = self._line1
            data['line1_entry_side'] = self._line1_entry_side
            data['polygon1'] = self._polygon1_points.copy()
            data['polygon1_finished'] = self._polygon1_finished
            if self._line1 and len(self._polygon1_points) >= 3 and self._polygon1_finished:
                data['complete'] = True
        
        return data
    
    def is_drawing_complete(self) -> bool:
        """Check if drawing is complete."""
        return self.get_drawing_data().get('complete', False)
    
    def _update_display(self):
        """Update the display with frame and drawings."""
        if self._current_frame is None:
            return
        
        frame = self._current_frame.copy()
        h, w = frame.shape[:2]
        
        # Draw shapes
        self._draw_shapes(frame)
        
        # Draw temp shape
        self._draw_temp_shape(frame)
        
        # Draw instructions
        self._draw_instructions(frame)
        
        # Convert to QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
    
    def _draw_shapes(self, frame: np.ndarray):
        """Draw all shapes on frame."""
        h, w = frame.shape[:2]
        
        # Draw line1
        if self._line1:
            p1 = self._normalized_to_pixels(self._line1[0], frame.shape)
            p2 = self._normalized_to_pixels(self._line1[1], frame.shape)
            color = (0, 255, 0)
            cv2.line(frame, p1, p2, color, 3)
            self._draw_direction_arrow(frame, p1, p2, self._line1_entry_side, color)
            
            if self._mode == DrawingMode.DUAL_LINE:
                cv2.putText(frame, "ENTRY GATE", (p1[0], p1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif self._mode == DrawingMode.LINE_AND_POLYGON:
                cv2.putText(frame, "GATE LINE", (p1[0], p1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw line2
        if self._line2:
            p1 = self._normalized_to_pixels(self._line2[0], frame.shape)
            p2 = self._normalized_to_pixels(self._line2[1], frame.shape)
            color = (0, 0, 255)
            cv2.line(frame, p1, p2, color, 3)
            self._draw_direction_arrow(frame, p1, p2, self._line2_entry_side, color)
            cv2.putText(frame, "EXIT GATE", (p1[0], p1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw polyline
        if len(self._polyline_points) >= 1:
            pts = [self._normalized_to_pixels(p, frame.shape) for p in self._polyline_points]
            
            # Use different color if finished
            color = (0, 255, 0) if self._polyline_finished else (0, 255, 255)
            
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], color, 3)
                
                # Draw direction arrow on each segment if finished
                if self._polyline_finished and len(pts) >= 2:
                    self._draw_direction_arrow(frame, pts[i], pts[i + 1], self._line1_entry_side, color)
            
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 8, color, -1)
                cv2.circle(frame, pt, 8, (255, 255, 255), 2)
                cv2.putText(frame, str(i + 1), (pt[0] - 5, pt[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        
        # Draw polygon1
        if len(self._polygon1_points) >= 1:
            pts = [self._normalized_to_pixels(p, frame.shape) for p in self._polygon1_points]
            pts_np = np.array(pts, np.int32)
            
            # Use different color if finished
            color = (0, 255, 0) if self._polygon1_finished else (255, 200, 0)
            
            if len(pts) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts_np], color)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.polylines(frame, [pts_np], True, color, 3)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(frame, pts[i], pts[i + 1], color, 3)
            
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 8, color, -1)
                cv2.circle(frame, pt, 8, (255, 255, 255), 2)
                cv2.putText(frame, str(i + 1), (pt[0] - 5, pt[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            
            if self._mode == DrawingMode.DUAL_POLYGON:
                cv2.putText(frame, "ZONE A", (pts[0][0], pts[0][1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif self._mode == DrawingMode.LINE_AND_POLYGON:
                cv2.putText(frame, "PARKING ZONE", (pts[0][0], pts[0][1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw polygon2
        if len(self._polygon2_points) >= 1:
            pts = [self._normalized_to_pixels(p, frame.shape) for p in self._polygon2_points]
            pts_np = np.array(pts, np.int32)
            
            color = (0, 255, 0) if self._polygon2_finished else (0, 200, 255)
            
            if len(pts) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts_np], color)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.polylines(frame, [pts_np], True, color, 3)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(frame, pts[i], pts[i + 1], color, 3)
            
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 8, color, -1)
                cv2.circle(frame, pt, 8, (255, 255, 255), 2)
                cv2.putText(frame, str(i + 1), (pt[0] - 5, pt[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            
            cv2.putText(frame, "ZONE B", (pts[0][0], pts[0][1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_temp_shape(self, frame: np.ndarray):
        """Draw temporary shape being drawn."""
        if self._temp_point is None:
            return
        
        temp_px = self._normalized_to_pixels(self._temp_point, frame.shape)
        
        # Crosshair
        cv2.line(frame, (temp_px[0] - 10, temp_px[1]), (temp_px[0] + 10, temp_px[1]), (255, 255, 255), 1)
        cv2.line(frame, (temp_px[0], temp_px[1] - 10), (temp_px[0], temp_px[1] + 10), (255, 255, 255), 1)
        
        if self._mode == DrawingMode.LINE:
            if self._is_drawing and self._drag_start:
                p1 = self._normalized_to_pixels(self._drag_start, frame.shape)
                cv2.line(frame, p1, temp_px, (255, 255, 0), 2)
        
        elif self._mode == DrawingMode.DUAL_LINE:
            if self._is_drawing and self._drag_start:
                p1 = self._normalized_to_pixels(self._drag_start, frame.shape)
                cv2.line(frame, p1, temp_px, (255, 255, 0), 2)
        
        elif self._mode == DrawingMode.POLYLINE:
            if self._polyline_points and not self._polyline_finished:
                last_pt = self._normalized_to_pixels(self._polyline_points[-1], frame.shape)
                cv2.line(frame, last_pt, temp_px, (255, 255, 0), 2)
        
        elif self._mode == DrawingMode.POLYGON:
            if self._polygon1_points and not self._polygon1_finished:
                last_pt = self._normalized_to_pixels(self._polygon1_points[-1], frame.shape)
                cv2.line(frame, last_pt, temp_px, (255, 255, 0), 2)
                first_pt = self._normalized_to_pixels(self._polygon1_points[0], frame.shape)
                cv2.line(frame, temp_px, first_pt, (255, 255, 0), 1)
        
        elif self._mode == DrawingMode.DUAL_POLYGON:
            if self._current_shape_index == 0 and self._polygon1_points and not self._polygon1_finished:
                last_pt = self._normalized_to_pixels(self._polygon1_points[-1], frame.shape)
                cv2.line(frame, last_pt, temp_px, (255, 255, 0), 2)
                first_pt = self._normalized_to_pixels(self._polygon1_points[0], frame.shape)
                cv2.line(frame, temp_px, first_pt, (255, 255, 0), 1)
            elif self._current_shape_index == 1 and self._polygon2_points and not self._polygon2_finished:
                last_pt = self._normalized_to_pixels(self._polygon2_points[-1], frame.shape)
                cv2.line(frame, last_pt, temp_px, (255, 255, 0), 2)
                first_pt = self._normalized_to_pixels(self._polygon2_points[0], frame.shape)
                cv2.line(frame, temp_px, first_pt, (255, 255, 0), 1)
        
        elif self._mode == DrawingMode.LINE_AND_POLYGON:
            if self._current_shape_index == 0:
                if self._is_drawing and self._drag_start:
                    p1 = self._normalized_to_pixels(self._drag_start, frame.shape)
                    cv2.line(frame, p1, temp_px, (255, 255, 0), 2)
            else:
                if self._polygon1_points and not self._polygon1_finished:
                    last_pt = self._normalized_to_pixels(self._polygon1_points[-1], frame.shape)
                    cv2.line(frame, last_pt, temp_px, (255, 255, 0), 2)
                    first_pt = self._normalized_to_pixels(self._polygon1_points[0], frame.shape)
                    cv2.line(frame, temp_px, first_pt, (255, 255, 0), 1)
    
    def _draw_instructions(self, frame: np.ndarray):
        """Draw on-screen instructions."""
        instructions = ""
        
        if self._mode == DrawingMode.LINE:
            if not self._line1:
                instructions = "Click and drag to draw a line"
            else:
                instructions = "Line complete!"
        
        elif self._mode == DrawingMode.POLYLINE:
            n = len(self._polyline_points)
            if self._polyline_finished:
                instructions = f"POLYLINE complete! ({n} points)"
            elif n == 0:
                instructions = "POLYLINE: Click to add points. Double-click to finish."
            else:
                instructions = f"POLYLINE: {n} points. Double-click or click Finish."
        
        elif self._mode == DrawingMode.DUAL_LINE:
            if not self._line1:
                instructions = "DUAL GATE: Draw ENTRY gate first (green)"
            elif not self._line2:
                instructions = "DUAL GATE: Now draw EXIT gate (red)"
            else:
                instructions = "DUAL GATE: Complete!"
        
        elif self._mode == DrawingMode.POLYGON:
            n = len(self._polygon1_points)
            if self._polygon1_finished:
                instructions = f"POLYGON complete! ({n} points)"
            elif n == 0:
                instructions = "POLYGON: Click to add points. Double-click to close."
            elif n < 3:
                instructions = f"POLYGON: {n} points. Need 3+."
            else:
                instructions = f"POLYGON: {n} points. Double-click or Finish to close."
        
        elif self._mode == DrawingMode.DUAL_POLYGON:
            n1 = len(self._polygon1_points)
            n2 = len(self._polygon2_points)
            if self._polygon1_finished and self._polygon2_finished:
                instructions = "Both zones complete!"
            elif not self._polygon1_finished:
                if n1 < 3:
                    instructions = f"ZONE A: {n1} points. Need 3+."
                else:
                    instructions = f"ZONE A: {n1} points. Double-click or Finish."
            else:
                if n2 < 3:
                    instructions = f"ZONE B: {n2} points. Need 3+."
                else:
                    instructions = f"ZONE B: {n2} points. Double-click or Finish."
        
        elif self._mode == DrawingMode.LINE_AND_POLYGON:
            n = len(self._polygon1_points)
            if self._current_shape_index == 0:
                if not self._line1:
                    instructions = "HYBRID: Step 1 - Draw gate line (click & drag)"
                else:
                    instructions = "HYBRID: Line done! Click 'Next Shape' below."
            else:
                if self._polygon1_finished:
                    instructions = f"HYBRID: Complete! ({n} points)"
                elif n < 3:
                    instructions = f"HYBRID: Parking zone {n} pts. Need 3+."
                else:
                    instructions = f"HYBRID: {n} points. Double-click or Finish."
        
        if instructions:
            (text_w, text_h), _ = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (10, 10), (20 + text_w, 35), (0, 0, 0), -1)
            cv2.putText(frame, instructions, (15, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_direction_arrow(self, frame: np.ndarray, p1: tuple, p2: tuple,
                               entry_side: str, color: tuple):
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
        
        offset = 30
        if entry_side == "side_a":
            start = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
            end = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))
        else:
            start = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))
            end = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
        
        cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.4)