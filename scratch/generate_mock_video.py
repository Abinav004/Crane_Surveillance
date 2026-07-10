import os
import cv2
import numpy as np

def create_mock_video(output_path: str, scenario: str = "compliance"):
    """
    Creates a synthetic 720p, 30fps video simulating a factory crane operation.
    
    Classes and color codes (BGR):
    - Person: Blue (255, 0, 0)
    - Crane Hook: Yellow (0, 255, 255)
    - Load Panel: Green (0, 255, 0)
    - Rubber Glove: Cyan (255, 255, 0)
    
    Scenarios:
    - "compliance": Worker wears gloves, attaches hook, and moves >30cm away.
    - "violations": Worker misses gloves for part of the time, and remains too close to the load.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    width, height = 1280, 720
    fps = 30
    duration_sec = 15
    total_frames = fps * duration_sec
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Static locations
    # Load Panel (Green)
    load_x1, load_y1, load_x2, load_y2 = 550, 450, 850, 600
    # Crane Hook (Yellow)
    hook_x1, hook_y1, hook_x2, hook_y2 = 670, 300, 730, 380
    
    for frame_idx in range(total_frames):
        # Create grey background (factory floor)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 80
        
        # Draw background grid lines to make it look like a factory floor
        for x in range(0, width, 100):
            cv2.line(frame, (x, 0), (x, height), (95, 95, 95), 1)
        for y in range(0, height, 100):
            cv2.line(frame, (0, y), (width, y), (95, 95, 95), 1)
            
        # Draw Load Panel
        cv2.rectangle(frame, (load_x1, load_y1), (load_x2, load_y2), (0, 255, 0), -1)
        cv2.putText(frame, "LOAD PANEL", (load_x1 + 10, load_y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        # Draw Crane Hook
        cv2.rectangle(frame, (hook_x1, hook_y1), (hook_x2, hook_y2), (0, 255, 255), -1)
        cv2.putText(frame, "HOOK", (hook_x1 + 5, hook_y1 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Calculate moving person (Blue) and hands/gloves (Cyan)
        # Sequence:
        # 0s - 3s (0-90 frames): Person enters from left, walks towards the load
        # 3s - 9s (90-270 frames): Person stands next to load, attaches chains (Attachment Phase)
        # 9s - 12s (270-360 frames): Person moves away
        # 12s - 15s (360-450 frames): Person stays at final position
        
        if frame_idx < 90:
            # Walking towards load
            progress = frame_idx / 90.0
            person_x1 = int(50 + progress * 400) # Ends at 450
            person_y1 = 300
            person_x2 = person_x1 + 100
            person_y2 = 650
            is_attaching = False
        elif frame_idx < 270:
            # Attaching hook
            person_x1 = 450
            person_y1 = 300
            person_x2 = 550
            person_y2 = 650
            is_attaching = True
        else:
            # Moving away
            progress = (frame_idx - 270) / 180.0 # From 9s to 15s
            if scenario == "compliance":
                # Moves far away (to x = 100)
                person_x1 = int(450 - progress * 350)
            elif scenario == "hook_zone_risk":
                # Moves to the right, directly under the hook (hook center is ~700)
                # Let's say worker center is 700, so worker x1 is 650 (width 100)
                person_x1 = int(450 + progress * 200)
            else:
                # Moves only slightly away (to x = 480), violating the 30cm (e.g. 100px) safety zone
                person_x1 = int(450 - progress * 30)
            person_y1 = 300
            person_x2 = person_x1 + 100
            person_y2 = 650
            is_attaching = False
            
        # Draw Person
        cv2.rectangle(frame, (person_x1, person_y1), (person_x2, person_y2), (255, 0, 0), -1)
        cv2.putText(frame, "WORKER", (person_x1 + 10, person_y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        # Draw Hands / Gloves (Cyan if wearing gloves, otherwise Skin color/Orange)
        if is_attaching:
            # Hands are raised near the hook and load panel overlap area
            hand_l_x1, hand_l_y1, hand_l_x2, hand_l_y2 = 580, 360, 615, 395
            hand_r_x1, hand_r_y1, hand_r_x2, hand_r_y2 = 625, 360, 660, 395
            
            # Check glove status based on scenario
            wear_gloves = True
            if scenario == "violations":
                # No gloves between frame 150 and 220 (approx 2.3 seconds)
                if 150 <= frame_idx <= 220:
                    wear_gloves = False
            
            glove_color = (255, 255, 0) if wear_gloves else (0, 165, 255) # Cyan if wearing, Orange if bare hands
            label = "GLOVE" if wear_gloves else "HAND"
            
            cv2.rectangle(frame, (hand_l_x1, hand_l_y1), (hand_l_x2, hand_l_y2), glove_color, -1)
            cv2.rectangle(frame, (hand_r_x1, hand_r_y1), (hand_r_x2, hand_r_y2), glove_color, -1)
            
            cv2.putText(frame, label, (hand_l_x1, hand_l_y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # Hands at the side
            hand_l_x1, hand_l_y1, hand_l_x2, hand_l_y2 = person_x1 - 15, 420, person_x1, 460
            hand_r_x1, hand_r_y1, hand_r_x2, hand_r_y2 = person_x2, 420, person_x2 + 15, 460
            # Just draw person body width covering them or simple rectangles
            cv2.rectangle(frame, (hand_l_x1, hand_l_y1), (hand_l_x2, hand_l_y2), (255, 0, 0), -1)
            cv2.rectangle(frame, (hand_r_x1, hand_r_y1), (hand_r_x2, hand_r_y2), (255, 0, 0), -1)

        # Draw frame number and scenario label for reference
        cv2.putText(frame, f"Frame: {frame_idx} | Scene: {scenario.upper()}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        out.write(frame)
        
    out.release()
    print(f"Mock video generated successfully: {output_path}")

if __name__ == "__main__":
    create_mock_video("data/raw_videos/scenario_compliance.mp4", "compliance")
    create_mock_video("data/raw_videos/scenario_violations.mp4", "violations")
    create_mock_video("data/raw_videos/scenario_hook_zone_risk.mp4", "hook_zone_risk")

