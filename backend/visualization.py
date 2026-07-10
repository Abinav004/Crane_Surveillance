import cv2
import numpy as np

def draw_danger_zone(frame: np.ndarray, coordinates: list, is_alert: bool) -> np.ndarray:
    """
    Draws the danger zone as a semi-transparent polygon.
    Outline is red, and fill is semi-transparent red.
    """
    if not coordinates:
        return frame
        
    pts = np.array(coordinates, dtype=np.int32).reshape((-1, 1, 2))
    
    # 1. Create semi-transparent overlay
    overlay = frame.copy()
    fill_color = (0, 0, 255) # Red in BGR
    cv2.fillPoly(overlay, [pts], fill_color)
    
    # Blend overlay with original frame
    # Higher opacity if alert is active
    alpha = 0.28 if is_alert else 0.15
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    
    # 2. Draw solid border outline
    # Red solid line
    border_color = (0, 0, 255)
    border_thickness = 3 if is_alert else 2
    cv2.polylines(frame, [pts], True, border_color, border_thickness)
    
    return frame

def draw_person_box(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float, 
                    track_id: int, is_inside: bool) -> np.ndarray:
    """
    Draws the person bounding box, foot contact point, and tracking ID label.
    Red if inside the danger zone (violating), Green if outside (safe).
    """
    color = (0, 0, 255) if is_inside else (0, 255, 0) # Red or Green
    thickness = 2
    
    # Draw rectangle bounding box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    # Label text (e.g., ID: 12)
    label = f"PERSON ID: {track_id}" if track_id is not None else "PERSON"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    
    # Get text size for rendering background label box
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Draw background box for text
    cv2.rectangle(frame, 
                  (int(x1), int(y1) - text_h - 10), 
                  (int(x1) + text_w, int(y1)), 
                  color, 
                  -1)
                  
    # Draw text
    cv2.putText(frame, 
                label, 
                (int(x1), int(y1) - 5), 
                font, 
                font_scale, 
                (255, 255, 255), 
                font_thickness, 
                cv2.LINE_AA)
                
    # Draw foot contact point (bottom center of bounding box)
    foot_x = (x1 + x2) / 2.0
    foot_y = y2
    cv2.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 255, 255), -1) # Yellow circle
    cv2.circle(frame, (int(foot_x), int(foot_y)), 6, color, 1)          # Outer colored ring
    
    return frame

def draw_alert_banner(frame: np.ndarray, width: int) -> np.ndarray:
    """
    Draws a prominent red banner overlay at the top of the frame to signify an intrusion.
    """
    banner_height = 50
    # Red filled rectangle at top
    cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 0, 255), -1)
    
    # Banner Text
    text = "ALERT: PERSON IN DANGER ZONE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 3
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = int((width - text_w) / 2)
    y = int((banner_height + text_h) / 2)
    
    # Draw text shadow (black) then text (white)
    cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame
