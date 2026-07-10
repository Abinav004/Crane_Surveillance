import os
import cv2
import pandas as pd
import logging
from pathlib import Path
from ultralytics import YOLO, YOLOWorld

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Class index definitions
# 0: person, 1: crane_hook, 2: load_panel, 3: rubber_glove
CLASS_NAMES = {
    0: "person",
    1: "crane_hook",
    2: "load_panel",
    3: "rubber_glove"
}

def track_objects_in_video(video_path: str, model_path: str, tracker_config: str = "bytetrack.yaml") -> pd.DataFrame:
    """
    Runs the YOLO model with ByteTrack on the video and outputs a DataFrame of tracked objects.
    
    DataFrame Columns:
        - frame_number (int)
        - timestamp_seconds (float)
        - track_id (int or None)
        - class_id (int)
        - class_name (str)
        - x1, y1, x2, y2 (float - pixel coordinates)
        - confidence (float)
    """
    logger.info(f"Loading tracking model from: {model_path}")
    if "world" in str(model_path).lower():
        logger.info("Using YOLO-World open-vocabulary tracking...")
        model = YOLOWorld(model_path)
        model.set_classes(["person", "crane hook", "machinery panel", "rubber glove"])
    else:
        model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return pd.DataFrame()
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0 # fallback default
    cap.release()
    
    logger.info(f"Running tracker on {video_path} at FPS={video_fps:.2f}...")
    
    # Run the track method which processes the video and tracks frames
    # persist=True ensures tracking is persistent across frames
    results = model.track(source=video_path, tracker=tracker_config, persist=True, verbose=False)
    
    tracking_records = []
    
    # Track ID history to detect switches for person class (class_id = 0)
    person_track_history = []  # list of (frame, track_id) for class_id 0
    active_person_ids = set()
    
    for frame_idx, res in enumerate(results):
        timestamp = frame_idx / video_fps
        boxes = res.boxes
        
        if boxes is None or len(boxes) == 0:
            continue
            
        # Extract individual box detections
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # ByteTrack will try to assign track IDs.
            # If tracking failed or model didn't track, box.id might be None.
            track_id = int(box.id[0].item()) if box.id is not None else None
            
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            x1, y1, x2, y2 = coords
            
            tracking_records.append({
                "frame_number": frame_idx,
                "timestamp_seconds": timestamp,
                "track_id": track_id,
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf
            })
            
            # Check for person track ID changes
            if cls_id == 0 and track_id is not None:
                person_track_history.append((frame_idx, track_id))
                active_person_ids.add(track_id)
                
    # Detect track ID switches for person (since there's typically only one worker in the video)
    # If the number of unique worker IDs is > 1 and they occur sequentially, warning is logged
    if len(active_person_ids) > 1:
        # Sort history by frame index
        person_track_history.sort(key=lambda x: x[0])
        # Find transition points
        last_id = None
        for frame, tid in person_track_history:
            if last_id is not None and tid != last_id:
                logger.warning(
                    f"TRACK SWITCH WARNING: Person track ID changed from {last_id} to {tid} "
                    f"at frame {frame} (approx. {frame/video_fps:.2f} seconds). "
                    "This may affect event duration and compliance calculations."
                )
            last_id = tid

    df = pd.DataFrame(tracking_records)
    logger.info(f"Tracking completed. Generated {len(df)} tracking records.")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run tracking on a video using trained weights and ByteTrack.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/crane_safety_yolo.pt", help="Path to trained YOLO weights")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        logger.error(f"Trained model not found at {args.model}. Please run train_yolo.py first.")
    else:
        df_track = track_objects_in_video(args.video, args.model)
        if not df_track.empty:
            print(df_track.head(20))
            print(f"Total rows: {len(df_track)}")
