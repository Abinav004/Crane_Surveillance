import cv2
import os
import logging

logger = logging.getLogger("frame_extraction")

def extract_first_frame(video_path: str, output_image_path: str) -> bool:
    """
    Extracts the first frame of a video and saves it as a JPEG.
    Returns True if successful, False otherwise.
    """
    logger.info(f"Extracting first frame from {video_path} to {output_image_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_path}")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error(f"Failed to read the first frame from {video_path}")
        return False
    
    # Ensure destination folder exists
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    # Save frame as JPEG
    success = cv2.imwrite(output_image_path, frame)
    if success:
        logger.info(f"Successfully saved first frame to {output_image_path}")
    else:
        logger.error(f"Failed to write image to {output_image_path}")
    return success

def get_video_info(video_path: str) -> dict:
    """
    Extracts metadata from the video including resolution, FPS, and frame count.
    """
    cap = cv2.VideoCapture(video_path)
    info = {"width": 0, "height": 0, "fps": 0.0, "frame_count": 0, "duration": 0.0}
    
    if cap.isOpened():
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
        info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
    
    cap.release()
    logger.info(f"Video info for {video_path}: {info}")
    return info
