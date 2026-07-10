import cv2
import numpy as np
import json
import os
import time
import logging
from ultralytics import YOLO
from backend.visualization import draw_danger_zone, draw_person_box, draw_alert_banner

logger = logging.getLogger("yolo_pipeline")

def run_intrusion_pipeline(
    video_path: str,
    zone_config: dict,
    output_video_path: str,
    output_log_path: str,
    progress_callback = None
):
    """
    Runs the YOLOv8 person tracking pipeline on a video.
    Checks frame-by-frame whether any tracked person's foot contact point overlaps the danger zone.
    Annotates the video frames with boxes/zones/banners and compiles the results.
    Generates a detailed intrusion events JSON log.
    """
    logger.info(f"Initiating YOLO tracking on video: {video_path}")
    logger.info(f"Target Zone: '{zone_config['zone_name']}' with coords: {zone_config['coordinates']}")
    
    # 1. Load YOLOv8 model (uses fine-tuned weights if available, otherwise base yolov8s.pt)
    model_path = "models/crane_safety_yolo.pt" if os.path.exists("models/crane_safety_yolo.pt") else "yolov8s.pt"
    logger.info(f"Using YOLO model weights from: {model_path}")
    model = YOLO(model_path)
    
    # Extract coordinates and compile into OpenCV polygon format
    coordinates = zone_config["coordinates"]
    zone_name = zone_config["zone_name"]
    zone_poly = np.array(coordinates, dtype=np.float32).reshape((-1, 1, 2))
    
    # 2. Open input video capture stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video source at {video_path}")
        raise FileNotFoundError(f"Video file not found or invalid: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        total_frames = 1
        
    logger.info(f"Video details: {width}x{height} @ {fps} FPS. Total frames: {total_frames}")
    
    # 3. Initialize video writer for annotated output
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Using mp4v codec for standard MP4 generation compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 4. State tracking variables
    events = []
    active_intrusions = {}  # track_id -> {"start_frame": int, "start_time": float}
    unique_intruder_ids = set()
    total_intrusion_frames = 0
    
    frame_idx = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Perform person tracking (class 0 is Person in COCO dataset)
            # persist=True maintains tracking IDs between successive frames
            # verbose=False suppresses frame logging to speed up terminal
            results = model.track(
                source=frame,
                persist=True,
                classes=[0],
                tracker="bytetrack.yaml",
                verbose=False
            )
            
            is_any_inside = False
            current_frame_intruder_ids = set()
            
            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Double-check class is person
                        cls = int(box.cls[0].item())
                        if cls != 0:
                            continue
                            
                        # Coordinates of box [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Bounding box track identifier
                        track_id = int(box.id[0].item()) if box.id is not None else None
                        
                        # Approximate person feet position using bottom center of the bounding box
                        # This avoids false alerts for overhead structures/crane heights
                        foot_x = (x1 + x2) / 2.0
                        foot_y = y2
                        
                        # Check proximity using pointPolygonTest (returns >= 0 if inside/on boundary)
                        is_inside = cv2.pointPolygonTest(zone_poly, (foot_x, foot_y), False) >= 0
                        
                        if is_inside:
                            is_any_inside = True
                            if track_id is not None:
                                current_frame_intruder_ids.add(track_id)
                                unique_intruder_ids.add(track_id)
                                
                        # Annotate individual person box (colored based on safety status)
                        frame = draw_person_box(frame, x1, y1, x2, y2, track_id, is_inside)
                        
            # State machine updates for intrusions
            # A. Process newly detected intrusions
            for track_id in current_frame_intruder_ids:
                if track_id not in active_intrusions:
                    active_intrusions[track_id] = {
                        "start_frame": frame_idx,
                        "start_time": frame_idx / fps
                    }
                    logger.warning(f"Intrusion start detected! Person ID: {track_id} at frame {frame_idx}")
                    
            # B. Process finished intrusions (persons who exited or vanished)
            finished_tracks = []
            for track_id, info in active_intrusions.items():
                if track_id not in current_frame_intruder_ids:
                    end_time = frame_idx / fps
                    events.append({
                        "track_id": track_id,
                        "start_frame": info["start_frame"],
                        "end_frame": frame_idx,
                        "start_time_sec": round(info["start_time"], 2),
                        "end_time_sec": round(end_time, 2),
                        "duration_sec": round(end_time - info["start_time"], 2)
                    })
                    finished_tracks.append(track_id)
                    logger.warning(f"Intrusion end detected! Person ID: {track_id} at frame {frame_idx}")
                    
            for track_id in finished_tracks:
                del active_intrusions[track_id]
                
            # C. Frame level visualization
            if is_any_inside:
                total_intrusion_frames += 1
                frame = draw_alert_banner(frame, width)
                
            # Draw the static danger zone perimeter
            frame = draw_danger_zone(frame, coordinates, is_any_inside)
            
            # Save frame to output stream
            out.write(frame)
            
            frame_idx += 1
            
            # Progress reporting callback
            if progress_callback and frame_idx % 10 == 0:
                prog = min(99, int((frame_idx / total_frames) * 100))
                progress_callback(prog)
                
    except Exception as e:
        logger.error(f"Error during video processing frame={frame_idx}: {e}")
        raise e
    finally:
        cap.release()
        out.release()
        
    # Handle active intrusions at video cutoff
    for track_id, info in active_intrusions.items():
        end_time = frame_idx / fps
        events.append({
            "track_id": track_id,
            "start_frame": info["start_frame"],
            "end_frame": frame_idx,
            "start_time_sec": round(info["start_time"], 2),
            "end_time_sec": round(end_time, 2),
            "duration_sec": round(end_time - info["start_time"], 2)
        })
        
    # Generate statistics summary
    total_intrusions = len(events)
    total_intrusion_time_sec = total_intrusion_frames / fps if fps > 0 else 0.0
    
    log_data = {
        "video_name": os.path.basename(video_path),
        "zone_name": zone_name,
        "summary": {
            "total_intrusions": total_intrusions,
            "total_intrusion_time_sec": round(total_intrusion_time_sec, 2),
            "unique_people_count": len(unique_intruder_ids),
            "processed_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "events": events
    }
    
    # Save statistics and events to log JSON
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    try:
        with open(output_log_path, "w") as f:
            json.dump(log_data, f, indent=4)
        logger.info(f"Intrusion event logs written to {output_log_path}")
    except Exception as e:
        logger.error(f"Failed to save event log JSON: {e}")
        
    if progress_callback:
        progress_callback(100)
        
    logger.info(f"YOLO pipeline complete for {video_path}. Total intrusions: {total_intrusions}")
    return log_data
