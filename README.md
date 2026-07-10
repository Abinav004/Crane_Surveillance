# Crane Operation Safety Monitoring Pipeline

This project implements a computer vision pipeline that automatically analyzes factory CCTV-style video footage of crane hook attachment operations. It evaluates:
1. **Safety Zone Compliance**: Ensures the worker moves at least 30cm (or a customized threshold) away from the load panel's bounding box after attaching the hook.
2. **Time-Efficiency**: Tracks the duration of the hook attachment window and flags operations that exceed 10 seconds.
3. **PPE Compliance**: Monitors whether rubber gloves are visibly worn on the worker's hands during the entire attachment window.

---

## Directory Structure

```text
├── data/
│   ├── raw_videos/          # Place input MP4 videos here to process
│   ├── extracted_frames/    # Auto-extracted video frames at target FPS
│   ├── labels/              # Auto-labeled YOLO-format annotation draft .txt files
│   │   └── previews/        # Downsized preview images with bounding boxes for spot-checking
│   └── dataset/             # Organized 80/20 train/val split for fine-tuning YOLO
├── models/
│   └── crane_safety_yolo.pt # Fine-tuned YOLO weights
├── src/
│   ├── frame_extraction.py  # Step 2: Extracts frames from MP4 videos
│   ├── auto_labeling.py     # Step 3: Open-vocabulary labeling with YOLO-World
│   ├── train_yolo.py        # Step 4: Split data and fine-tune YOLO model
│   ├── tracker.py           # Step 5: Bounding box tracking with ByteTrack
│   ├── rules_engine.py      # Step 6: Evaluates compliance parameters
│   ├── event_logger.py      # Step 7: Formats active violations into JSON reports
│   └── pipeline.py          # Step 8/9: Main CLI entry point and overlay visualizer
├── outputs/
│   ├── annotated_videos/    # Output video overlay with timers, readouts, and flashing alerts
│   └── event_logs/          # Chronological violation and summary JSON logs
├── requirements.txt         # Pinned python dependencies
└── README.md                # This manual
```

---

## Installation & Setup

1. **Python version**: Ensure you are running Python 3.10 or higher. (Verified on python 3.14+).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Add your videos**: Place your raw CCTV MP4 video files inside the `data/raw_videos/` directory.

---

## How to Run

The entire pipeline is orchestrated via `src/pipeline.py`.

### 1. Run Everything End-to-End
To extract frames, run auto-labeling, train the model, track, evaluate rules, write JSON logs, and render annotated videos for all raw footage:
```bash
python src/pipeline.py
```

### 2. Skip Long-running Tasks (e.g. Model Training)
If you already have a trained model at `models/crane_safety_yolo.pt` (or if you want the tracking script to fallback to a pre-trained base model like `yolov8s.pt` to test downstream logic immediately):
```bash
python src/pipeline.py --skip_training --skip_labeling --skip_extraction
```

### 3. Analyze a Specific Video
Run the pipeline on a single file outside the default directory:
```bash
python src/pipeline.py --video data/raw_videos/scenario_violations.mp4 --skip_training
```

---

## Calibration of Real-World Units

> [!WARNING]
> **Pixel-Based Distance Limitation**
> Without camera calibration, real-world metric measuring is highly approximate. The rules engine uses a default placeholder ratio (`--pixels_per_cm 2.0`). This conversion assumes that the worker and the load panel lie at a uniform distance relative to the camera lens and that there is no perspective distortion.

### How to Calibrate pixels-per-centimeter:
1. Identify a physical reference object of known real-world dimension visible in the camera view (e.g., the width of the load panel is 120cm, or the width of the crane hook is 15cm).
2. Open a video frame in an image editor and measure the pixel length of that reference object (e.g., if the 120cm load panel width measures 240 pixels in the frame).
3. Compute the scale factor:
   $$\text{scale\_factor} = \frac{\text{length in pixels}}{\text{real-world length in cm}} = \frac{240\text{ px}}{120\text{ cm}} = 2.0\text{ px/cm}$$
4. Supply your calibrated value when running the pipeline:
   ```bash
   python src/pipeline.py --pixels_per_cm 2.5
   ```

---

## Adding More Videos & Custom Classes

### To Add More Videos
Simply copy new `.mp4`, `.avi`, or `.mov` clips into [data/raw_videos/](file:///c:/Users/abinav/OneDrive/Desktop/Motivation%20app/data/raw_videos/) and run `python src/pipeline.py`. The frame extractor and trackers automatically loop through every video in the directory.

### To Add New Classes
1. Open [src/auto_labeling.py](file:///c:/Users/abinav/OneDrive/Desktop/Motivation%20app/src/auto_labeling.py).
2. Append your new class name (e.g., `"hard hat"`) to the `CLASS_NAMES` list.
3. Update [src/train_yolo.py](file:///c:/Users/abinav/OneDrive/Desktop/Motivation%20app/src/train_yolo.py) class configuration dictionary `names` and update number of classes `nc`.
4. Update the tracker mapping `CLASS_NAMES` in [src/tracker.py](file:///c:/Users/abinav/OneDrive/Desktop/Motivation%20app/src/tracker.py) and colors mapping in [src/pipeline.py](file:///c:/Users/abinav/OneDrive/Desktop/Motivation%20app/src/pipeline.py).
