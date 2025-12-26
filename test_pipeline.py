"""
Integration test for the people counting pipeline.
Tests: Video Source → YOLO Detector → Track Manager → Tripwire Engine

Usage:
    python test_pipeline.py                    # Use webcam
    python test_pipeline.py path/to/video.mp4  # Use video file
"""

import sys
import cv2
import time

from app.model.detector import YOLODetector, get_available_devices
from app.model.track_manager import TrackManager
from app.model.data_structures import (
    ModelConfig, 
    ROIConfiguration, 
    LineConfig,
    DebounceConfig,
)


class SimpleCounter:
    """
    Simple tripwire counter with clear logic.
    
    Line is VERTICAL in center of screen.
    - Person moves from LEFT to RIGHT → ENTRY
    - Person moves from RIGHT to LEFT → EXIT
    """
    
    def __init__(self, line_x: int, hysteresis: int = 20, cooldown: int = 30):
        self.line_x = line_x
        self.hysteresis = hysteresis
        self.cooldown = cooldown
        
        # Counts
        self.entry_count = 0
        self.exit_count = 0
        
        # Track states: {track_id: {"side": "left"/"right", "cooldown": int}}
        self.track_states = {}
    
    def get_side(self, x: float) -> str:
        """Determine which side of line the point is on."""
        if x < self.line_x - self.hysteresis:
            return "left"
        elif x > self.line_x + self.hysteresis:
            return "right"
        else:
            return "center"  # In hysteresis zone
    
    def process(self, detections: list) -> list:
        """
        Process detections and return crossing events.
        
        Returns list of: {"track_id": int, "direction": "entry"/"exit"}
        """
        events = []
        seen_ids = set()
        
        for det in detections:
            track_id = det.track_id
            if track_id < 0:
                continue
            
            seen_ids.add(track_id)
            
            # Get bottom-center x position
            x = det.bottom_center[0]
            current_side = self.get_side(x)
            
            # Skip if in hysteresis zone
            if current_side == "center":
                continue
            
            # Initialize new tracks
            if track_id not in self.track_states:
                self.track_states[track_id] = {
                    "side": current_side,
                    "cooldown": 0
                }
                continue
            
            state = self.track_states[track_id]
            
            # Decrease cooldown
            if state["cooldown"] > 0:
                state["cooldown"] -= 1
                continue
            
            previous_side = state["side"]
            
            # Check for crossing
            if previous_side != current_side:
                if previous_side == "left" and current_side == "right":
                    # LEFT → RIGHT = ENTRY
                    self.entry_count += 1
                    events.append({"track_id": track_id, "direction": "entry"})
                    state["cooldown"] = self.cooldown
                    
                elif previous_side == "right" and current_side == "left":
                    # RIGHT → LEFT = EXIT
                    self.exit_count += 1
                    events.append({"track_id": track_id, "direction": "exit"})
                    state["cooldown"] = self.cooldown
                
                # Update side
                state["side"] = current_side
        
        # Cleanup old tracks (not seen for a while)
        stale_ids = [tid for tid in self.track_states if tid not in seen_ids]
        for tid in stale_ids:
            # Keep for a bit longer in case they reappear
            pass
        
        return events
    
    def reset(self):
        """Reset counts and states."""
        self.entry_count = 0
        self.exit_count = 0
        self.track_states.clear()


def draw_frame(frame, detections, counter, line_x, width, height):
    """Draw all visualizations on frame."""
    
    # === Draw tripwire line ===
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 3)
    
    # === Draw hysteresis zone (shaded area) ===
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                  (line_x - counter.hysteresis, 0), 
                  (line_x + counter.hysteresis, height), 
                  (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    # === Draw side labels ===
    # LEFT side = ENTRY side (green)
    cv2.putText(frame, "ENTRY", (20, height // 2 - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, "SIDE", (20, height // 2 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.arrowedLine(frame, (80, height // 2 + 50), (line_x - 30, height // 2 + 50), 
                    (0, 255, 0), 3, tipLength=0.3)
    
    # RIGHT side = EXIT side (red)
    cv2.putText(frame, "EXIT", (width - 100, height // 2 - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(frame, "SIDE", (width - 100, height // 2 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.arrowedLine(frame, (width - 80, height // 2 + 50), (line_x + 30, height // 2 + 50), 
                    (0, 0, 255), 3, tipLength=0.3)
    
    # === Draw detections ===
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        bc_x, bc_y = int(det.bottom_center[0]), int(det.bottom_center[1])
        
        # Determine color based on side
        side = counter.get_side(bc_x)
        if side == "left":
            color = (0, 255, 0)  # Green = entry side
        elif side == "right":
            color = (0, 0, 255)  # Red = exit side
        else:
            color = (0, 255, 255)  # Yellow = in crossing zone
        
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Track ID
        if det.track_id >= 0:
            label = f"ID:{det.track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Bottom center point (this is what we track)
        cv2.circle(frame, (bc_x, bc_y), 8, color, -1)
        cv2.circle(frame, (bc_x, bc_y), 8, (255, 255, 255), 2)
    
    # === Draw counts panel ===
    panel_height = 120
    cv2.rectangle(frame, (0, 0), (180, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (180, panel_height), (255, 255, 255), 2)
    
    cv2.putText(frame, f"ENTRY: {counter.entry_count}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"EXIT:  {counter.exit_count}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    inside = counter.entry_count - counter.exit_count
    cv2.putText(frame, f"INSIDE: {inside}", (10, 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # === Instructions ===
    cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame


def main():
    # Determine source
    if len(sys.argv) > 1:
        source = sys.argv[1]
        source_type = "file" if not source.startswith("rtsp://") else "rtsp"
    else:
        source = 0
        source_type = "webcam"
    
    print("=" * 60)
    print("       PEOPLE COUNTING - PIPELINE TEST")
    print("=" * 60)
    print(f"Source: {source} ({source_type})")
    print()
    
    # Initialize detector
    print("Loading YOLO model (this may take a moment)...")
    config = ModelConfig(mode="cpu", model_size="n", confidence=0.5)
    detector = YOLODetector(config)
    
    if not detector.initialize():
        print("ERROR: Failed to initialize detector!")
        return
    
    print(f"Model: {detector.model_name} on {detector.device}")
    print()
    
    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Failed to open source: {source}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {width}x{height}")
    
    # Setup counter with line in center
    line_x = width // 2
    counter = SimpleCounter(line_x=line_x, hysteresis=25, cooldown=25)
    
    print()
    print("=" * 60)
    print("  HOW TO TEST:")
    print("  • Walk from LEFT to RIGHT → counts as ENTRY")
    print("  • Walk from RIGHT to LEFT → counts as EXIT")
    print("  • Yellow zone in center = crossing area")
    print("=" * 60)
    print()
    
    # FPS tracking
    fps_time = time.time()
    fps_frames = 0
    fps = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if source_type == "file":
                print("Video ended")
                break
            continue
        
        # Run detection with tracking
        detections = detector.track(frame)
        
        # Process counting
        events = counter.process(detections)
        
        # Print events
        for event in events:
            direction = event["direction"].upper()
            track_id = event["track_id"]
            symbol = "→→→" if direction == "ENTRY" else "←←←"
            print(f"  [{direction}] {symbol} Track {track_id}")
        
        # Draw visualization
        frame = draw_frame(frame, detections, counter, line_x, width, height)
        
        # FPS
        fps_frames += 1
        if fps_frames >= 20:
            fps = fps_frames / (time.time() - fps_time)
            fps_time = time.time()
            fps_frames = 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (width - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show
        cv2.imshow("People Counter Test", frame)
        
        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.reset()
            detector.reset_tracker()
            print("  [RESET] Counts cleared")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final results
    print()
    print("=" * 60)
    print("  FINAL RESULTS:")
    print(f"    Entry: {counter.entry_count}")
    print(f"    Exit:  {counter.exit_count}")
    print(f"    Inside: {counter.entry_count - counter.exit_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()