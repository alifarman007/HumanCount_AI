"""
YOLO Detector - Handles object detection and tracking using Ultralytics YOLOv11.
"""

import torch
from typing import List, Optional, Dict
from ultralytics import YOLO

from app.model.data_structures import Detection, ModelConfig


class YOLODetector:
    """
    YOLO-based person detector with built-in BoT-SORT tracking.
    
    Uses Ultralytics YOLOv11 for detection and tracking.
    Only detects people (class 0 in COCO dataset).
    """
    
    # COCO class ID for person
    PERSON_CLASS_ID = 0
    
    def __init__(self, config: ModelConfig = None):
        """
        Initialize detector.
        
        Args:
            config: Model configuration (defaults to CPU with nano model)
        """
        self._config = config or ModelConfig()
        self._model: Optional[YOLO] = None
        self._device: str = "cpu"
        self._initialized = False
        
        # Tracking persistence
        self._tracker = "botsort.yaml"  # Built-in BoT-SORT
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def model_name(self) -> str:
        return self._config.model_name
    
    def initialize(self) -> bool:
        """
        Load YOLO model.
        
        Returns:
            True if successful
        """
        try:
            # Determine device
            if self._config.mode == "gpu":
                if torch.cuda.is_available():
                    self._device = f"cuda:{self._config.device_id}"
                else:
                    print("CUDA not available, falling back to CPU")
                    self._device = "cpu"
            else:
                self._device = "cpu"
            
            # Load model
            model_name = self._config.model_name
            print(f"Loading {model_name} on {self._device}...")
            
            self._model = YOLO(model_name)
            self._model.to(self._device)
            
            # Warm up with dummy inference
            if self._device != "cpu":
                import numpy as np
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self._model.predict(dummy, verbose=False)
            
            self._initialized = True
            print(f"Model loaded successfully on {self._device}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            self._initialized = False
            return False
    
    def detect(self, frame) -> List[Detection]:
        """
        Run detection on a single frame (no tracking).
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            List of Detection objects
        """
        if not self._initialized:
            return []
        
        try:
            results = self._model.predict(
                frame,
                conf=self._config.confidence,
                classes=[self.PERSON_CLASS_ID],
                verbose=False
            )
            
            return self._parse_results(results)
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def track(self, frame) -> List[Detection]:
        """
        Run detection with tracking on a single frame.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            List of Detection objects with track IDs
        """
        if not self._initialized:
            return []
        
        try:
            results = self._model.track(
                frame,
                conf=self._config.confidence,
                classes=[self.PERSON_CLASS_ID],
                tracker=self._tracker,
                persist=True,  # Maintain track IDs across frames
                verbose=False
            )
            
            return self._parse_results(results, with_tracking=True)
            
        except Exception as e:
            print(f"Tracking error: {e}")
            return []
    
    def _parse_results(self, results, with_tracking: bool = False) -> List[Detection]:
        """Parse YOLO results into Detection objects."""
        detections = []
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]  # Single image, single result
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes
        
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            
            # Get track ID if tracking
            if with_tracking:
                if boxes.id is not None and i < len(boxes.id):
                    track_id = int(boxes.id[i].cpu().numpy())
                else:
                    # Skip detections without track ID (can't be tracked)
                    continue
            else:
                track_id = -1  # Detection-only mode
            
            # Skip invalid track IDs in tracking mode
            if with_tracking and track_id < 0:
                continue
            
            detection = Detection(
                track_id=track_id,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                confidence=conf,
                class_id=self.PERSON_CLASS_ID
            )
            
            detections.append(detection)
        
        return detections
    
    def reset_tracker(self):
        """Reset tracking state (clears all track IDs)."""
        if self._model:
            # Re-initialize model to reset tracker
            self._model.predictor = None
    
    def get_info(self) -> Dict:
        """Get detector information."""
        return {
            'model': self._config.model_name,
            'device': self._device,
            'initialized': self._initialized,
            'confidence': self._config.confidence,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


def get_available_devices() -> List[Dict]:
    """
    Get list of available compute devices.
    
    Returns:
        List of device info dicts
    """
    devices = [{'id': -1, 'name': 'CPU', 'type': 'cpu'}]
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append({
                'id': i,
                'name': f"GPU {i}: {name}",
                'type': 'cuda'
            })
    
    return devices