import os
import torch
from ultralytics import YOLOWorld

frame_path = "data/extracted_frames/The_worker_is_not_wearing_glov/frame_00020.jpg"

classes_options = [
    # Option 1
    ["person", "hook", "machinery", "glove"],
    # Option 2
    ["person", "crane hook", "metal panel", "gloved hand"],
    # Option 3
    ["person", "lifting hook", "load", "safety glove"],
    # Option 4
    ["person", "yellow hook", "generator", "yellow glove"],
    # Option 5
    ["person", "hoist hook", "heavy machinery", "work glove"],
    # Option 6
    ["person", "crane hook", "machinery panel", "rubber glove"]
]

for idx, classes in enumerate(classes_options):
    print(f"\n--- Testing Option {idx+1}: {classes} ---")
    # Reload model on CPU to avoid device mismatch bugs when resetting classes
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.to("cpu")
    model.set_classes(classes)
    
    results = model.predict(frame_path, conf=0.01, verbose=False)[0]
    
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            print(f"Class: {classes[cls_id]} ({cls_id}), Conf: {conf:.4f}, Box: {xyxy}")
    else:
        print("No detections found.")
