import os
import cv2
import argparse
import logging
from pathlib import Path
from ultralytics import YOLOWorld

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Class mapping:
# 0: person -> "person"
# 1: crane_hook -> "crane hook" or "hook"
# 2: load_panel -> "machinery panel" or "load panel"
# 3: rubber_glove -> "rubber glove" or "glove"
CLASS_NAMES = ["person", "crane hook", "machinery panel", "rubber glove"]

# Color mapping (BGR)
COLORS = {
    0: (255, 0, 0),      # Blue for person
    1: (0, 255, 255),    # Yellow for crane hook
    2: (0, 255, 0),      # Green for load panel
    3: (255, 255, 0)     # Cyan for rubber glove
}

def setup_yolo_world(model_name: str = "yolov8s-worldv2.pt") -> YOLOWorld:
    """
    Loads the YOLO-World open-vocabulary model and configures the offline custom classes.
    """
    logger.info(f"Loading YOLO-World model: {model_name}...")
    # YOLO-World is part of the ultralytics package and will download the checkpoint if not local
    model = YOLOWorld(model_name)
    
    # Define custom classes for open-vocabulary detection
    model.set_classes(CLASS_NAMES)
    logger.info(f"YOLO-World set to detect custom classes: {CLASS_NAMES}")
    return model

def run_auto_labeling(frames_dir: str, labels_dir: str, confidence_threshold: float = 0.3):
    """
    Processes all extracted frames, predicts boxes using YOLO-World, and generates draft labels.
    """
    frames_path = Path(frames_dir)
    labels_path = Path(labels_dir)
    previews_path = labels_path / "previews"
    
    previews_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize YOLO-World
    model = setup_yolo_world()
    
    # Track low confidence detections
    low_confidence_log_path = labels_path / "low_confidence_frames.txt"
    # Overwrite/Initialize log file
    with open(low_confidence_log_path, "w") as f:
        f.write("# List of frames with detection confidence < 0.5\n")

    # Find frame folders (one folder per video, skipping synthetic directory)
    video_dirs = [d for d in frames_path.iterdir() if d.is_dir() and d.name != "synthetic"]
    if not video_dirs:
        logger.warning(f"No extracted frame folders found in {frames_path}")
        return

    logger.info(f"Found {len(video_dirs)} video frame directory/directories to label.")
    
    for video_dir in video_dirs:
        video_name = video_dir.name
        video_labels_dir = labels_path / video_name
        video_labels_dir.mkdir(parents=True, exist_ok=True)
        
        video_previews_dir = previews_path / video_name
        video_previews_dir.mkdir(parents=True, exist_ok=True)
        
        frame_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        logger.info(f"Processing {len(frame_files)} frames for video: {video_name}")
        
        for frame_file in frame_files:
            img = cv2.imread(str(frame_file))
            if img is None:
                continue
                
            img_height, img_width = img.shape[:2]
            
            # Predict using YOLO-World
            results = model.predict(frame_file, conf=confidence_threshold, verbose=False)[0]
            
            yolo_labels = []
            has_low_confidence = False
            
            # Create a copy for previews
            preview_img = img.copy()
            
            # Parse predictions
            if results.boxes is not None:
                for box in results.boxes:
                    # Convert to tensor/numpy float
                    coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    if conf < 0.5:
                        has_low_confidence = True
                    
                    # Convert coordinates to YOLO format: class_id x_center y_center width height (normalized)
                    x1, y1, x2, y2 = coords
                    box_w = x2 - x1
                    box_h = y2 - y1
                    x_center = x1 + (box_w / 2.0)
                    y_center = y1 + (box_h / 2.0)
                    
                    # Normalize
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    w_norm = box_w / img_width
                    h_norm = box_h / img_height
                    
                    yolo_labels.append(f"{cls_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                    
                    # Draw on preview image
                    color = COLORS.get(cls_id, (255, 255, 255))
                    cv2.rectangle(preview_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label_text = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"
                    cv2.putText(preview_img, label_text, (int(x1), int(y1) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save labels to file (.txt)
            label_filename = frame_file.stem + ".txt"
            label_file_path = video_labels_dir / label_filename
            with open(label_file_path, "w") as f:
                f.write("\n".join(yolo_labels) + "\n")
                
            # Save low-res preview image (scaled down to width 640px to save storage)
            preview_scale = 640 / img_width
            preview_h = int(img_height * preview_scale)
            preview_resized = cv2.resize(preview_img, (640, preview_h))
            preview_path_file = video_previews_dir / frame_file.name
            cv2.imwrite(str(preview_path_file), preview_resized)
            
            # Log low confidence detections
            if has_low_confidence or len(yolo_labels) == 0:
                # Flag frame if confidence < 0.5 or if nothing detected (likely needs manual annotation)
                with open(low_confidence_log_path, "a") as f:
                    reason = "low_conf_detections" if has_low_confidence else "no_detections"
                    f.write(f"{frame_file.relative_to(frames_path.parent)} | {reason}\n")
                    
        logger.info(f"Finished auto-labeling video: {video_name}")
    logger.info(f"Auto-labeling complete. Low-confidence logs written to: {low_confidence_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label frames using open-vocabulary detection (YOLO-World).")
    parser.add_argument("--frames_dir", type=str, default="data/extracted_frames", help="Path to extracted frames folder")
    parser.add_argument("--labels_dir", type=str, default="data/labels", help="Path to save output labels and previews")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO-World detection")
    
    args = parser.parse_args()
    run_auto_labeling(args.frames_dir, args.labels_dir, args.conf)
