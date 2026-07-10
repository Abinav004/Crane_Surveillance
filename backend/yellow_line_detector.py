import cv2
import numpy as np
import os
import logging

logger = logging.getLogger("yellow_line_detector")

def suggest_danger_zone(image_path: str, margin_px: int = 30) -> list:
    """
    Suggests a candidate danger zone box based on yellow floor markings.
    
    Logic:
    1. Loads the first frame image.
    2. Converts to HSV color space.
    3. Thresholds for standard industrial yellow markings (typical range: H 20-35).
       *NOTE: HSV values depend heavily on lighting, camera sensor, and site-specific
       factors. In a real-world deployment, these thresholds should be configurable per camera.
    4. Applies Morphological Close & Open to bridge broken lines and eliminate speckle noise.
    5. Finds all contours, filters by minimum area to ignore reflections/dust, and
       extracts the overall boundary (union) of all major yellow contours.
    6. Expands the combined bounding box outward by margin_px (which acts as a 30cm equivalent safety buffer).
    7. Returns the 4 corner points in clockwise order: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]].
    
    If no yellow markings are found, returns None.
    """
    logger.info(f"Running yellow floor-marking detection on {image_path} with margin={margin_px}px")
    
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read image at {image_path}")
        return None
        
    h, w, _ = img.shape
    
    # 1. Convert to HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Define HSV range for industrial yellow floor-markings
    # H: 15-38 (roughly 30-76 degrees on the hue circle, matching yellow/orange-yellow)
    # S: 80-255 (moderate to high saturation)
    # V: 80-255 (moderate to high brightness/value)
    # In practice, these values should be adjusted for the camera's lighting profile.
    lower_yellow = np.array([15, 80, 80], dtype=np.uint8)
    upper_yellow = np.array([38, 255, 255], dtype=np.uint8)
    
    # Create binary mask
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 3. Clean up the mask using morphological operations
    # Close: bridges gaps in dashed or faded lines
    # Open: removes small speckle noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. Find contours on the cleaned binary mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Minimum area threshold (e.g. 200 pixels) to avoid small artifacts/reflections
    min_area = 200
    significant_points = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            significant_points.extend(cnt)
            
    # 5. If no markings are found, return None
    if not significant_points:
        logger.warning("No significant yellow markings detected in the frame.")
        return None
        
    # 6. Compute bounding rectangle around all significant yellow marking contours combined
    pts_array = np.array(significant_points)
    x, y, box_w, box_h = cv2.boundingRect(pts_array)
    
    # 7. Expand the bounding box outward by the safety margin offset
    # Clamping coordinates to remain inside frame boundaries
    x1 = max(0, x - margin_px)
    y1 = max(0, y - margin_px)
    x2 = min(w - 1, x + box_w + margin_px)
    y2 = min(h - 1, y + box_h + margin_px)
    
    # Formulate coordinate output [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (as a clockwise quad)
    suggested_coords = [
        [float(x1), float(y1)], # Top-left
        [float(x2), float(y1)], # Top-right
        [float(x2), float(y2)], # Bottom-right
        [float(x1), float(y2)]  # Bottom-left
    ]
    
    logger.info(f"Floor marking bounding box detected. Suggested coords: {suggested_coords}")
    return suggested_coords
