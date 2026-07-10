import os
import cv2
import argparse
import logging
import pandas as pd
from pathlib import Path

# Import our custom modules
from frame_extraction import process_all_videos
from auto_labeling import run_auto_labeling
from train_yolo import prepare_dataset, train_yolo
from tracker import track_objects_in_video
from rules_engine import RulesEngine, box_edge_distance
from event_logger import create_event_log

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Class and color config
CLASS_NAMES = {0: "person", 1: "crane_hook", 2: "load_panel", 3: "rubber_glove"}
COLORS = {
    0: (255, 0, 0),      # Blue for person
    1: (0, 255, 255),    # Yellow for crane_hook
    2: (0, 255, 0),      # Green for load_panel
    3: (255, 255, 0)     # Cyan for rubber_glove
}

def annotate_video(video_path: str, tracking_df: pd.DataFrame, rules_result: dict, 
                   pixels_per_cm: float, safety_dist_cm: float, output_path: str):
    """
    Renders an annotated output video with bounding boxes, timers, distance readouts, 
    and flashing red alerts overlaid on active violations.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file to annotate: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.info(f"Rendering annotated video: {output_path} ({width}x{height} @ {fps:.2f}fps)")
    
    start_frame = rules_result.get("attachment_start_frame")
    end_frame = rules_result.get("attachment_end_frame")
    start_time = rules_result.get("attachment_start_time", 0.0)
    
    ppe_frames = set(rules_result.get("ppe_violation_frames", []))
    time_frames = set(rules_result.get("time_violation_frames", []))
    safety_frames = set(rules_result.get("safety_zone_violation_frames", []))
    hook_zone_frames = set(rules_result.get("hook_zone_violation_frames", []))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_idx / fps
        
        # 1. Retrieve and draw tracking bounding boxes for this frame
        frame_tracks = tracking_df[tracking_df["frame_number"] == frame_idx]
        
        # Draw the Hook Floor Danger Zone Column
        hook_zone_width_px = 100.0
        hook_tracks = frame_tracks[frame_tracks["class_id"] == 1]
        if not hook_tracks.empty:
            h_row = hook_tracks.iloc[0]
            hx1, hy1, hx2, hy2 = int(h_row["x1"]), int(h_row["y1"]), int(h_row["x2"]), int(h_row["y2"])
            hcx = (hx1 + hx2) / 2.0
            zx1 = int(hcx - hook_zone_width_px / 2.0)
            zx2 = int(hcx + hook_zone_width_px / 2.0)
            
            is_risk = frame_idx in hook_zone_frames
            zone_color = (0, 0, 255) if is_risk else (0, 255, 255)
            
            # Draw semi-transparent rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (zx1, hy2), (zx2, height), zone_color, -1)
            alpha = 0.2
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
            
            # Draw boundaries
            cv2.line(frame, (zx1, hy2), (zx1, height), zone_color, 2, cv2.LINE_AA)
            cv2.line(frame, (zx2, hy2), (zx2, height), zone_color, 2, cv2.LINE_AA)
            
        for _, track in frame_tracks.iterrows():
            cls_id = int(track["class_id"])
            x1, y1, x2, y2 = int(track["x1"]), int(track["y1"]), int(track["x2"]), int(track["y2"])
            track_id = int(track["track_id"]) if pd.notna(track["track_id"]) else None
            
            color = COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{CLASS_NAMES.get(cls_id, f'class_{cls_id}')}"
            if track_id is not None:
                label += f" ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 2. Determine state and draw UI layers (timer, distance, alerts)
        state_text = "STATE: PRE-ATTACHMENT"
        timer_text = ""
        dist_text = ""
        active_violations = []
        
        if start_frame is not None and frame_idx >= start_frame:
            if frame_idx <= end_frame:
                state_text = "STATE: ATTACHMENT IN PROGRESS"
                elapsed = timestamp - start_time
                timer_text = f"Attachment Time: {elapsed:.1f}s"
                
                # Check for active PPE and Time violations
                if frame_idx in ppe_frames:
                    active_violations.append("PPE VIOLATION (NO GLOVES)")
                if frame_idx in time_frames:
                    active_violations.append("TIME LIMIT EXCEEDED")
            else:
                state_text = "STATE: POST-ATTACHMENT"
                timer_text = f"Attachment Duration: {rules_result.get('duration_seconds', 0.0):.1f}s"
                
                # Calculate real-time distance readout
                person_boxes = frame_tracks[frame_tracks["class_id"] == 0]
                load_boxes = frame_tracks[frame_tracks["class_id"] == 2]
                
                if not person_boxes.empty and not load_boxes.empty:
                    p_box = (person_boxes.iloc[0]["x1"], person_boxes.iloc[0]["y1"], person_boxes.iloc[0]["x2"], person_boxes.iloc[0]["y2"])
                    l_box = (load_boxes.iloc[0]["x1"], load_boxes.iloc[0]["y1"], load_boxes.iloc[0]["x2"], load_boxes.iloc[0]["y2"])
                    dist_px = box_edge_distance(p_box, l_box)
                    dist_cm = dist_px / pixels_per_cm
                    dist_text = f"Worker-Load Distance: {dist_cm:.1f} cm"
                else:
                    dist_text = "Worker-Load Distance: N/A"
                    
                # Check for active safety zone violations
                if frame_idx in safety_frames:
                    active_violations.append("SAFETY ZONE VIOLATION")
                    
        # Check for active hook floor zone risk (at any time)
        if frame_idx in hook_zone_frames:
            active_violations.append("RISK: WORKER UNDER HOOK")

        # 3. Draw UI texts
        # Semantics layout box at the top left
        cv2.rectangle(frame, (10, 10), (450, 100), (0, 0, 0), -1)
        cv2.putText(frame, state_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if timer_text:
            cv2.putText(frame, timer_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if dist_text:
            cv2.putText(frame, dist_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 4. Trigger alert overlays if violations are active
        if active_violations:
            # Flashing border logic (3 frames ON, 3 frames OFF)
            if (frame_idx % 6) < 3:
                # Draw thick red boundary border
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
                
            # Draw alert banner on screen
            banner_y = 120
            for violation in active_violations:
                cv2.rectangle(frame, (10, banner_y), (400, banner_y + 30), (0, 0, 255), -1)
                cv2.putText(frame, f"ALERT: {violation}", (20, banner_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                banner_y += 40

        # Draw Overall Compliance Status at the bottom right
        cv2.rectangle(frame, (width - 320, height - 140), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.putText(frame, "COMPLIANCE STATUS", (width - 300, height - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        ppe_status = "PPE: COMPLIANT" if not rules_result.get("ppe_violation") else "PPE: VIOLATION"
        time_status = "TIME: COMPLIANT" if not rules_result.get("time_violation") else "TIME: VIOLATION"
        safe_status = "SAFETY ZONE: SAFE" if not rules_result.get("safety_zone_violation") else "SAFETY ZONE: VIOLATION"
        hook_status = "HOOK ZONE: SAFE" if not rules_result.get("hook_zone_violation") else "HOOK ZONE: RISK"
        
        ppe_col = (0, 255, 0) if not rules_result.get("ppe_violation") else (0, 0, 255)
        time_col = (0, 255, 0) if not rules_result.get("time_violation") else (0, 0, 255)
        safe_col = (0, 255, 0) if not rules_result.get("safety_zone_violation") else (0, 0, 255)
        hook_col = (0, 255, 0) if not rules_result.get("hook_zone_violation") else (0, 0, 255)
        
        cv2.putText(frame, ppe_status, (width - 300, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ppe_col, 1)
        cv2.putText(frame, time_status, (width - 300, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, time_col, 1)
        cv2.putText(frame, safe_status, (width - 300, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, safe_col, 1)
        cv2.putText(frame, hook_status, (width - 300, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hook_col, 1)

        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    logger.info(f"Finished rendering annotated video to: {output_path}")

def run_pipeline(video_path_str: str, skip_extraction: bool, skip_labeling: bool, 
                 skip_training: bool, model_path_str: str, training_epochs: int, 
                 training_batch: int, pixels_per_cm: float, safety_dist_cm: float,
                 attachment_time_sec: float, glove_tolerance_pct: float,
                 near_touch_threshold: float):
    """
    Orchestrates the end-to-end processing pipeline.
    """
    logger.info("Initializing Pipeline Execution...")
    
    # Paths setup
    raw_videos_dir = "data/raw_videos"
    extracted_frames_dir = "data/extracted_frames"
    labels_dir = "data/labels"
    dataset_dir = "data/dataset"
    models_dir = "models"
    output_videos_dir = "outputs/annotated_videos"
    output_logs_dir = "outputs/event_logs"
    
    # 1. Scrape videos to process
    video_path = Path(video_path_str)
    video_files = []
    if video_path.is_file():
        video_files.append(video_path)
    elif video_path.is_dir():
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        video_files = sorted([f for f in video_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions])
    else:
        # Check if default path exists
        default_p = Path(raw_videos_dir)
        if default_p.exists():
            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            video_files = sorted([f for f in default_p.iterdir() if f.is_file() and f.suffix.lower() in video_extensions])
            
    if not video_files:
        logger.error(f"No video files found for path: {video_path_str}")
        return
        
    logger.info(f"Targeting {len(video_files)} video(s) for pipeline analysis.")

    # 2. Step 2: Frame Extraction
    if not skip_extraction:
        logger.info("--- Starting Step 2: Frame Extraction ---")
        process_all_videos(raw_videos_dir, extracted_frames_dir, target_fps=5.0)
    else:
        logger.info("Bypassing Step 2 (Frame Extraction).")

    # 3. Step 3: Auto-Labeling
    if not skip_labeling:
        logger.info("--- Starting Step 3: Auto-Labeling (Zero-Shot) ---")
        run_auto_labeling(extracted_frames_dir, labels_dir, confidence_threshold=0.25)
    else:
        logger.info("Bypassing Step 3 (Auto-Labeling).")

    # 4. Step 4: YOLO Fine-Tuning
    final_model_path = Path(model_path_str)
    if not skip_training:
        try:
            yaml_path = prepare_dataset(extracted_frames_dir, labels_dir, dataset_dir)
            train_yolo(yaml_path, "yolov8s.pt", training_epochs, training_batch, models_dir)
        except Exception as e:
            logger.error(f"YOLO training failed: {e}. Downstream pipeline will fallback to zero-shot YOLO-World model.")
            
    # Ensure the model exists, otherwise fallback to open-vocabulary YOLO-World
    if not final_model_path.exists():
        logger.warning(f"Model weight file {final_model_path} not found. Downstream tracker will fallback to zero-shot YOLO-World model: yolov8s-worldv2.pt.")
        final_model_path = Path("yolov8s-worldv2.pt")
    else:
        logger.info(f"Using trained model weights: {final_model_path}")

    # Instantiate Rules Engine
    # CRITICAL: Note about pixels_per_cm conversion inside comments/logs
    logger.info(f"Configuring Rules Engine: safety_dist={safety_dist_cm}cm, limit={attachment_time_sec}s, scale={pixels_per_cm}px/cm (placeholder calibration)")
    rules_engine = RulesEngine(
        near_touch_threshold_px=near_touch_threshold,
        pixels_per_cm=pixels_per_cm,
        safety_distance_threshold_cm=safety_dist_cm,
        max_attachment_duration_sec=attachment_time_sec,
        ppe_missing_tolerance_pct=glove_tolerance_pct
    )

    # 5. Process each video through tracking, rules engine, event logging, and visualization
    for video_file in video_files:
        video_name = video_file.name
        logger.info(f"==================================================")
        logger.info(f"Processing Video pipeline for: {video_name}")
        logger.info(f"==================================================")
        
        # Open source to get metadata
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            fps = 30.0

        # Step 5: Run Tracking
        logger.info(f"--- Step 5: Object Tracking (ByteTrack) ---")
        tracking_df = track_objects_in_video(str(video_file), str(final_model_path))
        
        if tracking_df.empty:
            logger.warning(f"No tracked entities captured for {video_name}. Skipping rules evaluation.")
            continue

        # Step 6: Rules Evaluation
        logger.info(f"--- Step 6: Evaluating Safety Rules ---")
        rules_result = rules_engine.evaluate_video_rules(video_name, tracking_df)
        
        # Step 7: Event Logging
        logger.info(f"--- Step 7: Event Logging ---")
        log_filepath = create_event_log(rules_result, fps, output_logs_dir)
        
        # Step 8: Visualization
        logger.info(f"--- Step 8: Rendering Annotated Video ---")
        annotated_video_path = os.path.join(output_videos_dir, f"{video_file.stem}_annotated.mp4")
        annotate_video(str(video_file), tracking_df, rules_result, pixels_per_cm, safety_dist_cm, annotated_video_path)
        
        logger.info(f"Successfully processed video: {video_name}!")
        
    logger.info("Pipeline end-to-end execution complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end Crane Operation Safety Monitoring Pipeline.")
    parser.add_argument("--video", type=str, default="data/raw_videos", help="Path to input video file or folder containing videos")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip Step 2: Frame Extraction")
    parser.add_argument("--skip_labeling", action="store_true", help="Skip Step 3: Auto-Labeling zero-shot")
    parser.add_argument("--train", action="store_true", help="Run Step 4: YOLO Fine-Tuning (disabled by default)")
    parser.add_argument("--model", type=str, default="models/crane_safety_yolo.pt", help="Path to YOLO weights file")
    
    # Hyperparams / Threshold args
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for training YOLO model")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training YOLO model")
    parser.add_argument("--pixels_per_cm", type=float, default=2.0, help="Approximate scaling factor from pixels to real-world cm (calibration)")
    parser.add_argument("--safety_dist", type=float, default=30.0, help="Required minimum safety distance in cm")
    parser.add_argument("--attachment_time", type=float, default=10.0, help="Configurable attachment threshold in seconds")
    parser.add_argument("--glove_tolerance", type=float, default=20.0, help="Percent margin of error for missing rubber gloves")
    parser.add_argument("--near_touch_threshold", type=float, default=150.0, help="Pixel distance threshold to trigger attachment start (default: 150.0)")
    
    args = parser.parse_args()
    
    # Run the pipeline orchestration
    run_pipeline(
        video_path_str=args.video,
        skip_extraction=args.skip_extraction,
        skip_labeling=args.skip_labeling,
        skip_training=not args.train,
        model_path_str=args.model,
        training_epochs=args.epochs,
        training_batch=args.batch,
        pixels_per_cm=args.pixels_per_cm,
        safety_dist_cm=args.safety_dist,
        attachment_time_sec=args.attachment_time,
        glove_tolerance_pct=args.glove_tolerance,
        near_touch_threshold=args.near_touch_threshold
    )
