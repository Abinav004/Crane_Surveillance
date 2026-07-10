import os
import cv2
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_frames(video_path: Path, output_dir: Path, target_fps: float) -> int:
    """
    Extracts frames from a video file at a specified FPS.
    
    Args:
        video_path (Path): Path to the input video file.
        output_dir (Path): Path to the directory where extracted frames will be saved.
        target_fps (float): Desired frames per second to extract.
        
    Returns:
        int: Number of frames extracted.
    """
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 0

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return 0

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    
    logger.info(f"Processing: {video_path.name}")
    logger.info(f"Video properties: FPS={video_fps:.2f}, Total Frames={total_frames}, Duration={duration_sec:.2f}s")

    # If the video's actual FPS is lower than target_fps, extract all frames
    if video_fps <= 0:
        logger.warning("Could not determine video FPS. Defaulting to extracting every frame.")
        step = 1
    else:
        step = max(1, round(video_fps / target_fps))
        logger.info(f"Target extraction rate: {target_fps} FPS (Extracting every {step} frames)")

    # Ensure output directory exists for this video
    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we should save this frame
        if frame_count % step == 0:
            # Format filename with leading zeros (e.g. frame_00001.jpg) to ensure sorting order
            out_filename = f"frame_{saved_count:05d}.jpg"
            out_path = video_output_dir / out_filename
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    logger.info(f"Successfully extracted {saved_count} frames to {video_output_dir}\n")
    return saved_count

def process_all_videos(input_dir: str, output_dir: str, target_fps: float):
    """
    Scans the input directory for video files and extracts frames from each of them.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    # Look for common video file extensions
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]

    if not video_files:
        logger.warning(f"No video files found in {input_path}")
        return

    logger.info(f"Found {len(video_files)} video(s) to process.")
    
    total_saved = 0
    for video_file in video_files:
        total_saved += extract_frames(video_file, output_path, target_fps)
        
    logger.info(f"Frame extraction complete. Extracted {total_saved} total frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video files at a configurable FPS.")
    parser.add_argument("--input_dir", type=str, default="data/raw_videos", help="Path to raw input videos folder")
    parser.add_argument("--output_dir", type=str, default="data/extracted_frames", help="Path to save extracted frames")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second to extract (default: 5.0)")
    
    args = parser.parse_args()
    process_all_videos(args.input_dir, args.output_dir, args.fps)
