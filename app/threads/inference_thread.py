"""
Inference Thread - Runs YOLO detection and counting engine in separate thread.
Processes frames from queue, emits results for UI display.
"""

from typing import Optional, Dict, List
from queue import Queue, Empty
from PySide6.QtCore import QThread, Signal, QMutex

from app.model.detector import YOLODetector
from app.model.track_manager import TrackManager
from app.model.data_structures import Detection, ModelConfig, ROIConfiguration, CountEvent
from app.engines.base import CountingEngineBase


class InferenceThread(QThread):
    """
    Thread for running YOLO inference and counting logic.
    
    Receives frames via queue, processes them, and emits results.
    Decoupled from video capture for smooth display.
    
    Signals:
        inference_ready: Emits (frame, detections, counts, frame_number)
        inference_error: Emits error message
        model_loaded: Emits when model is ready
    """
    
    # Signals
    inference_ready = Signal(object, list, dict, int)  # frame, detections, counts, frame_num
    inference_error = Signal(str)
    model_loaded = Signal(bool, str)  # success, message
    count_event = Signal(object)  # CountEvent
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Components
        self._detector: Optional[YOLODetector] = None
        self._track_manager: Optional[TrackManager] = None
        self._engine: Optional[CountingEngineBase] = None
        
        # Frame queue
        self._frame_queue: Queue = Queue(maxsize=30)
        
        # State
        self._running = False
        self._model_config: Optional[ModelConfig] = None
        self._roi_config: Optional[ROIConfiguration] = None
        
        # Mutex for thread-safe config updates
        self._mutex = QMutex()
        
        # Stats
        self._processed_frames = 0
        self._skipped_frames = 0
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def processed_frames(self) -> int:
        return self._processed_frames
    
    @property
    def queue_size(self) -> int:
        return self._frame_queue.qsize()
    
    def configure(
        self,
        model_config: ModelConfig,
        roi_config: ROIConfiguration
    ):
        """
        Configure the inference thread.
        
        Args:
            model_config: YOLO model configuration
            roi_config: ROI and counting engine configuration
        """
        self._mutex.lock()
        self._model_config = model_config
        self._roi_config = roi_config
        self._mutex.unlock()
    
    def set_roi_config(self, roi_config: ROIConfiguration):
        """Update ROI configuration (can be called while running)."""
        self._mutex.lock()
        self._roi_config = roi_config
        
        # Recreate engine with new config
        if roi_config:
            try:
                self._engine = CountingEngineBase.create_engine(
                    roi_config.module_type,
                    roi_config
                )
            except Exception as e:
                self.inference_error.emit(f"Failed to create engine: {e}")
        
        self._mutex.unlock()
    
    def enqueue_frame(self, frame, frame_number: int):
        """
        Add frame to processing queue.
        
        Args:
            frame: BGR image (numpy array)
            frame_number: Frame sequence number
        """
        if not self._running:
            return
        
        # Drop oldest frame if queue is full (keeps display smooth)
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
                self._skipped_frames += 1
            except Empty:
                pass
        
        try:
            self._frame_queue.put_nowait((frame.copy(), frame_number))
        except:
            pass
    
    def run(self):
        """Main inference loop."""
        # Initialize detector
        if not self._model_config:
            self.inference_error.emit("No model configuration provided")
            return
        
        self._detector = YOLODetector(self._model_config)
        if not self._detector.initialize():
            self.model_loaded.emit(False, "Failed to load YOLO model")
            return
        
        self.model_loaded.emit(True, f"Model loaded on {self._detector.device}")
        
        # Initialize track manager
        self._track_manager = TrackManager(
            history_length=30,
            stale_threshold=90,
            position_type="bottom_center"
        )
        
        # Initialize counting engine
        self._mutex.lock()
        if self._roi_config:
            try:
                self._engine = CountingEngineBase.create_engine(
                    self._roi_config.module_type,
                    self._roi_config
                )
            except Exception as e:
                self.inference_error.emit(f"Failed to create engine: {e}")
        self._mutex.unlock()
        
        self._running = True
        self._processed_frames = 0
        self._skipped_frames = 0
        
        while self._running:
            try:
                # Get frame from queue (with timeout to check running flag)
                frame, frame_number = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue
            
            # Run detection with tracking
            detections = self._detector.track(frame)
            
            # Update track manager
            tracks = self._track_manager.update(detections)
            
            # Run counting engine
            counts = {}
            self._mutex.lock()
            if self._engine:
                events = self._engine.process_tracks(tracks, self._track_manager)
                counts = self._engine.counts
                
                # Emit count events
                for event in events:
                    self.count_event.emit(event)
            self._mutex.unlock()
            
            self._processed_frames += 1
            
            # Emit results
            self.inference_ready.emit(frame, detections, counts, frame_number)
        
        # Cleanup
        if self._engine:
            self._engine.cleanup()
    
    def stop(self):
        """Stop inference thread."""
        self._running = False
        
        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break
        
        # Wait for thread to finish
        self.wait(5000)
    
    def reset_counts(self):
        """Reset counting engine counts."""
        self._mutex.lock()
        if self._engine:
            self._engine.reset_counts()
        self._mutex.unlock()
    
    def reset_tracker(self):
        """Reset tracker and track manager."""
        if self._detector:
            self._detector.reset_tracker()
        if self._track_manager:
            self._track_manager.reset()
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            'processed_frames': self._processed_frames,
            'skipped_frames': self._skipped_frames,
            'queue_size': self._frame_queue.qsize(),
            'track_count': self._track_manager.active_track_count if self._track_manager else 0
        }