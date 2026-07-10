import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def group_consecutive_frames(frames: List[int]) -> List[List[int]]:
    """
    Groups a list of frame numbers into sublists of consecutive frame sequences.
    Example: [1, 2, 3, 10, 11, 15] -> [[1, 2, 3], [10, 11], [15]]
    """
    if not frames:
        return []
    
    sorted_frames = sorted(frames)
    groups = [[sorted_frames[0]]]
    
    for f in sorted_frames[1:]:
        if f == groups[-1][-1] + 1:
            groups[-1].append(f)
        else:
            groups.append([f])
            
    return groups

def create_event_log(rules_result: Dict[str, Any], fps: float, output_dir: str = "outputs/event_logs") -> str:
    """
    Translates rule engine evaluations into grouped event logs and exports to a JSON file.
    """
    video_name = rules_result["video_name"]
    os.makedirs(output_dir, exist_ok=True)
    
    events = []
    
    # 1. Process PPE compliance violations
    if rules_result["ppe_violation"]:
        ppe_groups = group_consecutive_frames(rules_result["ppe_violation_frames"])
        for group in ppe_groups:
            start_f = group[0]
            end_f = group[-1]
            start_t = start_f / fps
            end_t = end_f / fps
            events.append({
                "type": "PPE_VIOLATION",
                "start_frame": start_f,
                "end_frame": end_f,
                "start_time_sec": round(start_t, 2),
                "end_time_sec": round(end_t, 2),
                "details": f"Rubber gloves absent during portion of attachment window (missing in {len(group)} frame(s))."
            })
            
    # 2. Process Time efficiency violations
    if rules_result["time_violation"]:
        time_groups = group_consecutive_frames(rules_result["time_violation_frames"])
        for group in time_groups:
            start_f = group[0]
            end_f = group[-1]
            start_t = start_f / fps
            end_t = end_f / fps
            events.append({
                "type": "TIME_VIOLATION",
                "start_frame": start_f,
                "end_frame": end_f,
                "start_time_sec": round(start_t, 2),
                "end_time_sec": round(end_t, 2),
                "details": f"Attachment window took {rules_result['duration_seconds']:.2f}s, exceeding 10.0s threshold."
            })
            
    # 3. Process Safety zone compliance violations
    if rules_result["safety_zone_violation"]:
        safety_groups = group_consecutive_frames(rules_result["safety_zone_violation_frames"])
        for group in safety_groups:
            start_f = group[0]
            end_f = group[-1]
            start_t = start_f / fps
            end_t = end_f / fps
            events.append({
                "type": "SAFETY_ZONE_VIOLATION",
                "start_frame": start_f,
                "end_frame": end_f,
                "start_time_sec": round(start_t, 2),
                "end_time_sec": round(end_t, 2),
                "details": f"Worker remained closer than 30cm safe distance proxy after attachment (min distance: {rules_result['min_distance_cm_estimate']:.1f}cm)."
            })

    # 4. Process Hook floor zone compliance violations
    if rules_result.get("hook_zone_violation"):
        hook_groups = group_consecutive_frames(rules_result["hook_zone_violation_frames"])
        for group in hook_groups:
            start_f = group[0]
            end_f = group[-1]
            start_t = start_f / fps
            end_t = end_f / fps
            events.append({
                "type": "HOOK_ZONE_VIOLATION",
                "start_frame": start_f,
                "end_frame": end_f,
                "start_time_sec": round(start_t, 2),
                "end_time_sec": round(end_t, 2),
                "details": f"Worker detected standing or walking directly under the crane hook (frame range {start_f}-{end_f})."
            })

    # Build the summary stats
    summary = {
        "total_violations": len(events),
        "ppe_compliant": not rules_result["ppe_violation"],
        "time_efficient": not rules_result["time_violation"],
        "safety_zone_compliant": not rules_result["safety_zone_violation"],
        "hook_zone_compliant": not rules_result.get("hook_zone_violation"),
        "attachment_duration_seconds": round(rules_result["duration_seconds"], 2),
        "min_distance_cm_estimate": round(rules_result["min_distance_cm_estimate"], 2),
        "max_distance_cm_estimate": round(rules_result["max_distance_cm_estimate"], 2)
    }
    
    log_data = {
        "video": video_name,
        "timestamp_analyzed": datetime.now().isoformat(),
        "events": events,
        "summary": summary
    }
    
    # Save to file
    video_stem = Path(video_name).stem
    log_filename = f"{video_stem}_log.json"
    log_filepath = os.path.join(output_dir, log_filename)
    
    with open(log_filepath, "w") as f:
        json.dump(log_data, f, indent=2)
        
    logger.info(f"Saved event log for {video_name} to {log_filepath}")
    return log_filepath

if __name__ == "__main__":
    # Test logger with mock rules output
    mock_result = {
        "video_name": "scenario_violations.mp4",
        "attachment_start_frame": 90,
        "attachment_end_frame": 270,
        "attachment_start_time": 3.0,
        "attachment_end_time": 9.0,
        "duration_seconds": 6.0,
        "min_distance_cm_estimate": 15.2,
        "max_distance_cm_estimate": 25.4,
        "ppe_violation": True,
        "time_violation": False,
        "safety_zone_violation": True,
        "ppe_violation_frames": list(range(150, 221)),
        "time_violation_frames": [],
        "safety_zone_violation_frames": list(range(271, 330))
    }
    
    log_path = create_event_log(mock_result, fps=30.0)
    print(f"Mock log successfully created. File size: {os.path.getsize(log_path)} bytes")
