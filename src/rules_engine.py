import math
import logging
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def box_edge_distance(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """
    Computes the minimum edge-to-edge Euclidean distance between two bounding boxes A and B.
    Each box is defined as (x1, y1, x2, y2).
    
    If they overlap, the distance is 0.
    """
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    
    # Calculate horizontal gap
    dist_x = max(0.0, bx1 - ax2, ax1 - bx2)
    # Calculate vertical gap
    dist_y = max(0.0, by1 - ay2, ay1 - by2)
    
    # Euclidean distance of the gaps
    return math.sqrt(dist_x**2 + dist_y**2)

def get_centroid(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Calculates the center point (x, y) of a bounding box.
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def estimate_hand_region(person_box: Tuple[float, float, float, float], 
                           hook_box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Approximates the worker's hand region as the upper portion of the person's box
    that faces the hook box.
    """
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = hook_box
    
    # Upper portion of person box (top 40%)
    hand_y1 = py1
    hand_y2 = py1 + 0.4 * (py2 - py1)
    
    # Determine horizontal side based on hook's position relative to person's centroid
    person_cx = (px1 + px2) / 2.0
    hook_cx = (hx1 + hx2) / 2.0
    
    if hook_cx > person_cx:
        # Hook is to the right, hand is likely on the right side of the worker's torso
        hand_x1 = px1 + 0.2 * (px2 - px1)
        hand_x2 = px2
    else:
        # Hook is to the left, hand is likely on the left side of the worker's torso
        hand_x1 = px1
        hand_x2 = px1 + 0.8 * (px2 - px1)
        
    return (hand_x1, hand_y1, hand_x2, hand_y2)

def boxes_intersect(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> bool:
    """
    Returns True if boxA and boxB overlap in any way.
    """
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

class RulesEngine:
    def __init__(self, 
                 near_touch_threshold_px: float = 80.0,
                 consecutive_frames_end: int = 5,
                 pixels_per_cm: float = 2.0,         # CRITICAL: Rough approximation placeholder. Calibrate in prod!
                 safety_distance_threshold_cm: float = 30.0,
                 max_attachment_duration_sec: float = 10.0,
                 ppe_missing_tolerance_pct: float = 20.0,
                 post_attachment_window_sec: float = 10.0,
                 hook_floor_zone_width_px: float = 100.0):
        
        self.near_touch_threshold_px = near_touch_threshold_px
        self.consecutive_frames_end = consecutive_frames_end
        self.pixels_per_cm = pixels_per_cm
        self.safety_distance_threshold_cm = safety_distance_threshold_cm
        self.max_attachment_duration_sec = max_attachment_duration_sec
        self.ppe_missing_tolerance_pct = ppe_missing_tolerance_pct
        self.post_attachment_window_sec = post_attachment_window_sec
        self.hook_floor_zone_width_px = hook_floor_zone_width_px

    def evaluate_video_rules(self, video_name: str, tracking_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Processes tracked object detections frame-by-frame and evaluates crane safety rules.
        """
        if tracking_df.empty:
            logger.warning(f"No tracking data provided for video: {video_name}")
            return self._empty_result(video_name)

        # Get unique frames
        frames = sorted(tracking_df["frame_number"].unique())
        
        # Step (e): Crane Hook Floor Zone Violation Check (evaluated for ALL frames)
        hook_zone_violation_frames = []
        for frame in frames:
            frame_df = tracking_df[tracking_df["frame_number"] == frame]
            person_boxes = frame_df[frame_df["class_id"] == 0]
            hook_boxes = frame_df[frame_df["class_id"] == 1]
            
            if not person_boxes.empty and not hook_boxes.empty:
                p_box = (person_boxes.iloc[0]["x1"], person_boxes.iloc[0]["y1"], person_boxes.iloc[0]["x2"], person_boxes.iloc[0]["y2"])
                h_box = (hook_boxes.iloc[0]["x1"], hook_boxes.iloc[0]["y1"], hook_boxes.iloc[0]["x2"], hook_boxes.iloc[0]["y2"])
                
                hcx = (h_box[0] + h_box[2]) / 2.0
                zx1 = hcx - self.hook_floor_zone_width_px / 2.0
                zx2 = hcx + self.hook_floor_zone_width_px / 2.0
                
                px1, py1, px2, py2 = p_box
                hy2 = h_box[3]
                
                # Overlaps horizontally and person is below the bottom of the crane hook
                if px1 <= zx2 and px2 >= zx1 and py2 >= hy2:
                    hook_zone_violation_frames.append(frame)
        
        # Determine overall video FPS
        total_frames = len(frames)
        if total_frames > 1:
            frame_diffs = pd.Series(frames).diff().dropna()
            # Find the mean time difference if timestamp exists
            timestamps = [tracking_df[tracking_df["frame_number"] == f]["timestamp_seconds"].iloc[0] for f in frames]
            time_diffs = pd.Series(timestamps).diff().dropna()
            mean_time_diff = time_diffs.mean()
            video_fps = 1.0 / mean_time_diff if mean_time_diff > 0 else 30.0
        else:
            video_fps = 30.0

        # Step (a): Detect Attachment Window
        attachment_start_frame = None
        attachment_end_frame = None
        
        # Keep track of centroid distances between person and load_panel
        centroid_distances = {} # frame_number -> distance
        
        # 1. Detect start
        for frame in frames:
            frame_df = tracking_df[tracking_df["frame_number"] == frame]
            
            # Filter boxes
            person_boxes = frame_df[frame_df["class_id"] == 0]
            hook_boxes = frame_df[frame_df["class_id"] == 1]
            load_boxes = frame_df[frame_df["class_id"] == 2]
            
            if person_boxes.empty or hook_boxes.empty or load_boxes.empty:
                continue
                
            p_box = (person_boxes.iloc[0]["x1"], person_boxes.iloc[0]["y1"], person_boxes.iloc[0]["x2"], person_boxes.iloc[0]["y2"])
            h_box = (hook_boxes.iloc[0]["x1"], hook_boxes.iloc[0]["y1"], hook_boxes.iloc[0]["x2"], hook_boxes.iloc[0]["y2"])
            l_box = (load_boxes.iloc[0]["x1"], load_boxes.iloc[0]["y1"], load_boxes.iloc[0]["x2"], load_boxes.iloc[0]["y2"])
            
            # Start: person near hook AND person near load panel
            dist_p_h = box_edge_distance(p_box, h_box)
            dist_p_l = box_edge_distance(p_box, l_box)
            
            if dist_p_h <= self.near_touch_threshold_px and dist_p_l <= self.near_touch_threshold_px:
                attachment_start_frame = frame
                logger.info(f"[{video_name}] Attachment start detected at Frame {frame} (Time: {frame/video_fps:.2f}s)")
                break

        # If attachment never started, we cannot evaluate the rest
        if attachment_start_frame is None:
            logger.warning(f"[{video_name}] Attachment start never detected.")
            return self._empty_result(video_name, hook_zone_violation_frames)

        # 2. Detect end (consecutive frames of moving away)
        # We start searching from attachment_start_frame
        moving_away_counter = 0
        prev_dist = None
        
        for frame in range(attachment_start_frame, max(frames) + 1):
            frame_df = tracking_df[tracking_df["frame_number"] == frame]
            person_boxes = frame_df[frame_df["class_id"] == 0]
            load_boxes = frame_df[frame_df["class_id"] == 2]
            
            if person_boxes.empty or load_boxes.empty:
                # If they disappear, reset counter or ignore
                continue
                
            p_box = (person_boxes.iloc[0]["x1"], person_boxes.iloc[0]["y1"], person_boxes.iloc[0]["x2"], person_boxes.iloc[0]["y2"])
            l_box = (load_boxes.iloc[0]["x1"], load_boxes.iloc[0]["y1"], load_boxes.iloc[0]["x2"], load_boxes.iloc[0]["y2"])
            
            p_c = get_centroid(p_box)
            l_c = get_centroid(l_box)
            
            # Centroid distance
            dist = math.sqrt((p_c[0] - l_c[0])**2 + (p_c[1] - l_c[1])**2)
            centroid_distances[frame] = dist
            
            if prev_dist is not None:
                if dist > prev_dist:
                    moving_away_counter += 1
                else:
                    moving_away_counter = 0 # reset if not moving away
                    
                if moving_away_counter >= self.consecutive_frames_end:
                    # Attachment end is the frame when the movement away started (t - N)
                    attachment_end_frame = frame - self.consecutive_frames_end
                    logger.info(f"[{video_name}] Attachment end detected at Frame {attachment_end_frame} (Time: {attachment_end_frame/video_fps:.2f}s)")
                    break
            prev_dist = dist

        # Fallback if worker never moved away
        if attachment_end_frame is None:
            attachment_end_frame = max(frames)
            logger.info(f"[{video_name}] Worker never moved away. Defaulting end frame to last frame: {attachment_end_frame}")

        # Compute timestamps
        start_time = tracking_df[tracking_df["frame_number"] == attachment_start_frame]["timestamp_seconds"].iloc[0]
        end_time = tracking_df[tracking_df["frame_number"] == attachment_end_frame]["timestamp_seconds"].iloc[0]
        duration_seconds = end_time - start_time

        # Step (c): Time violation check
        time_violation = duration_seconds > self.max_attachment_duration_sec
        time_violation_frames = []
        if time_violation:
            # Violation begins after max_attachment_duration_sec has elapsed from start
            violation_start_time = start_time + self.max_attachment_duration_sec
            for frame in range(attachment_start_frame, attachment_end_frame + 1):
                f_time = tracking_df[tracking_df["frame_number"] == frame]["timestamp_seconds"].iloc[0] if not tracking_df[tracking_df["frame_number"] == frame].empty else frame/video_fps
                if f_time > violation_start_time:
                    time_violation_frames.append(frame)

        # Step (d): PPE Check (within attachment window)
        total_window_frames = 0
        glove_present_frames = 0
        ppe_violation_frames = []

        for frame in range(attachment_start_frame, attachment_end_frame + 1):
            frame_df = tracking_df[tracking_df["frame_number"] == frame]
            person_boxes = frame_df[frame_df["class_id"] == 0]
            hook_boxes = frame_df[frame_df["class_id"] == 1]
            glove_boxes = frame_df[frame_df["class_id"] == 3]
            
            if person_boxes.empty or hook_boxes.empty:
                continue
                
            total_window_frames += 1
            
            p_box = (person_boxes.iloc[0]["x1"], person_boxes.iloc[0]["y1"], person_boxes.iloc[0]["x2"], person_boxes.iloc[0]["y2"])
            h_box = (hook_boxes.iloc[0]["x1"], hook_boxes.iloc[0]["y1"], hook_boxes.iloc[0]["x2"], hook_boxes.iloc[0]["y2"])
            
            # Approximate hand region
            hand_region = estimate_hand_region(p_box, h_box)
            
            # Check if any glove box overlaps the estimated hand region
            glove_detected = False
            for _, g_row in glove_boxes.iterrows():
                g_box = (g_row["x1"], g_row["y1"], g_row["x2"], g_row["y2"])
                if boxes_intersect(hand_region, g_box):
                    glove_detected = True
                    break
                    
            if glove_detected:
                glove_present_frames += 1
            else:
                ppe_violation_frames.append(frame)

        # Check if missing gloves percentage exceeds threshold (tolerance is 20%, i.e. 80% wearing required)
        if total_window_frames > 0:
            missing_pct = ((total_window_frames - glove_present_frames) / total_window_frames) * 100.0
            ppe_violation = missing_pct > self.ppe_missing_tolerance_pct
        else:
            ppe_violation = False
            missing_pct = 0.0

        # Step (b): Safety zone compliance check
        # Look at post-attachment window: 10 seconds after attachment end
        post_window_frames = []
        post_window_end_time = end_time + self.post_attachment_window_sec
        
        for frame in range(attachment_end_frame + 1, max(frames) + 1):
            f_rows = tracking_df[tracking_df["frame_number"] == frame]
            if f_rows.empty:
                continue
            f_time = f_rows["timestamp_seconds"].iloc[0]
            if f_time <= post_window_end_time:
                post_window_frames.append(frame)

        # Compute minimum edge-to-edge distance in the post-attachment window
        min_distance_px = float('inf')
        safety_zone_violation_frames = []
        
        # Safety threshold in pixels
        safety_threshold_px = self.safety_distance_threshold_cm * self.pixels_per_cm
        
        for frame in post_window_frames:
            frame_df = tracking_df[tracking_df["frame_number"] == frame]
            person_boxes = frame_df[frame_df["class_id"] == 0]
            load_boxes = frame_df[frame_df["class_id"] == 2]
            
            if person_boxes.empty or load_boxes.empty:
                continue
                
            p_box = (person_boxes.iloc[0]["x1"], person_boxes.iloc[0]["y1"], person_boxes.iloc[0]["x2"], person_boxes.iloc[0]["y2"])
            l_box = (load_boxes.iloc[0]["x1"], load_boxes.iloc[0]["y1"], load_boxes.iloc[0]["x2"], load_boxes.iloc[0]["y2"])
            
            dist_px = box_edge_distance(p_box, l_box)
            if dist_px < min_distance_px:
                min_distance_px = dist_px
                
            # If distance is below safety zone threshold, flag frame
            if dist_px < safety_threshold_px:
                safety_zone_violation_frames.append(frame)

        # If no post-attachment frames were captured, min_distance_px remains inf
        if min_distance_px == float('inf'):
            # Fallback to the distance at the end of attachment
            min_distance_px = 0.0
            
        min_distance_cm = min_distance_px / self.pixels_per_cm
        
        # SAFETY ZONE VIOLATION: Worker remains closer than the threshold for the ENTIRE post-attachment window
        # Meaning, they never exceeded the safe distance. If they stayed close (violating in every post frame), or
        # if the minimum distance achieved during the window was below the threshold.
        # Wait, the prompt says: "Flag SAFETY_ZONE_VIOLATION if the worker never exceeds this distance within the post-attachment window".
        # This means that at NO POINT in the post-attachment window did the worker get further than 30cm.
        # i.e., in ALL frames, they were closer than 30cm, which is equivalent to: maximum distance in the window < threshold.
        # Let's compute the max distance in the window to verify. If max distance achieved is < threshold, they never exceeded it.
        max_distance_px = 0.0
        for frame in post_window_frames:
            frame_df = tracking_df[tracking_df["frame_number"] == frame]
            person_boxes = frame_df[frame_df["class_id"] == 0]
            load_boxes = frame_df[frame_df["class_id"] == 2]
            if person_boxes.empty or load_boxes.empty:
                continue
            p_box = (person_boxes.iloc[0]["x1"], person_boxes.iloc[0]["y1"], person_boxes.iloc[0]["x2"], person_boxes.iloc[0]["y2"])
            l_box = (load_boxes.iloc[0]["x1"], load_boxes.iloc[0]["y1"], load_boxes.iloc[0]["x2"], load_boxes.iloc[0]["y2"])
            dist_px = box_edge_distance(p_box, l_box)
            if dist_px > max_distance_px:
                max_distance_px = dist_px
                
        max_distance_cm = max_distance_px / self.pixels_per_cm
        
        if len(post_window_frames) > 0:
            # Never exceeded safety threshold
            safety_zone_violation = max_distance_cm < self.safety_distance_threshold_cm
        else:
            # If no post-window frame exists (video ended immediately), check distance at the end frame
            safety_zone_violation = min_distance_cm < self.safety_distance_threshold_cm

        return {
            "video_name": video_name,
            "attachment_start_frame": attachment_start_frame,
            "attachment_end_frame": attachment_end_frame,
            "attachment_start_time": start_time,
            "attachment_end_time": end_time,
            "duration_seconds": duration_seconds,
            "min_distance_cm_estimate": min_distance_cm,
            "max_distance_cm_estimate": max_distance_cm if len(post_window_frames) > 0 else min_distance_cm,
            "ppe_violation": ppe_violation,
            "time_violation": time_violation,
            "safety_zone_violation": safety_zone_violation,
            "hook_zone_violation": len(hook_zone_violation_frames) > 0,
            "ppe_violation_frames": ppe_violation_frames if ppe_violation else [],
            "time_violation_frames": time_violation_frames if time_violation else [],
            "safety_zone_violation_frames": safety_zone_violation_frames if safety_zone_violation else [],
            "hook_zone_violation_frames": hook_zone_violation_frames
        }

    def _empty_result(self, video_name: str, hook_zone_violation_frames: List[int] = None) -> Dict[str, Any]:
        if hook_zone_violation_frames is None:
            hook_zone_violation_frames = []
        return {
            "video_name": video_name,
            "attachment_start_frame": None,
            "attachment_end_frame": None,
            "attachment_start_time": 0.0,
            "attachment_end_time": 0.0,
            "duration_seconds": 0.0,
            "min_distance_cm_estimate": 0.0,
            "max_distance_cm_estimate": 0.0,
            "ppe_violation": False,
            "time_violation": False,
            "safety_zone_violation": False,
            "hook_zone_violation": len(hook_zone_violation_frames) > 0,
            "ppe_violation_frames": [],
            "time_violation_frames": [],
            "safety_zone_violation_frames": [],
            "hook_zone_violation_frames": hook_zone_violation_frames
        }
