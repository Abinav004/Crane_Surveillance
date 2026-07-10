import os
import shutil
import logging
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Import our backend modules
from backend.frame_extraction import extract_first_frame, get_video_info
from backend.zone_config import save_zone_config, load_zone_config
from backend.yellow_line_detector import suggest_danger_zone
from backend.yolo_pipeline import run_intrusion_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backend_main")

app = FastAPI(title="Danger-Zone Intrusion Calibration & Processing Server")

# Allow CORS for development ease
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
os.makedirs("data/raw_videos", exist_ok=True)
os.makedirs("config/danger_zones", exist_ok=True)
os.makedirs("outputs/annotated_videos", exist_ok=True)
os.makedirs("outputs/event_logs", exist_ok=True)
os.makedirs("frontend", exist_ok=True)

# Mount static directories
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# In-memory dictionary to track processing tasks
# video_name -> status dict
processing_tasks = {}

class ZoneSaveRequest(BaseModel):
    video_name: str
    zone_name: str
    coordinates: List[List[float]]  # List of [x, y] coordinates

@app.get("/")
def read_root():
    """
    Serves the calibration UI HTML page.
    """
    calibration_path = "frontend/calibration.html"
    if os.path.exists(calibration_path):
        return FileResponse(calibration_path)
    return JSONResponse(status_code=404, content={"message": f"calibration.html not found at {calibration_path}"})

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Receives a video file, saves it to data/uploads/, 
    extracts the first frame, and returns video properties.
    """
    logger.info(f"Received video upload request: {file.filename}")
    video_path = os.path.join("data/raw_videos", file.filename)
    
    # Save the uploaded file in chunks
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Video saved successfully at {video_path}")
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    # Extract first frame
    first_frame_filename = f"{os.path.splitext(file.filename)[0]}_first.jpg"
    first_frame_path = os.path.join("data/raw_videos", first_frame_filename)
    
    success = extract_first_frame(video_path, first_frame_path)
    if not success:
        logger.error("Failed to extract the first frame of the video")
        raise HTTPException(status_code=500, detail="Failed to extract first frame from video.")
        
    info = get_video_info(video_path)
    
    return {
        "video_name": file.filename,
        "first_frame_url": f"/data/raw_videos/{first_frame_filename}",
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "frame_count": info["frame_count"],
        "duration": info["duration"]
    }

@app.get("/api/videos")
def list_videos():
    """
    Returns a list of all uploaded video filenames.
    """
    videos = []
    uploads_dir = "data/raw_videos"
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                videos.append(f)
    return videos

@app.get("/api/video-info/{video_name}")
def get_info(video_name: str):
    """
    Returns details of an already uploaded video, and extracts first frame if missing.
    """
    video_path = os.path.join("data/raw_videos", video_name)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")
        
    first_frame_filename = f"{os.path.splitext(video_name)[0]}_first.jpg"
    first_frame_path = os.path.join("data/raw_videos", first_frame_filename)
    
    if not os.path.exists(first_frame_path):
        extract_first_frame(video_path, first_frame_path)
        
    info = get_video_info(video_path)
    return {
        "video_name": video_name,
        "first_frame_url": f"/data/raw_videos/{first_frame_filename}",
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "frame_count": info["frame_count"],
        "duration": info["duration"]
    }

@app.get("/api/suggest-zone/{video_name}")
def suggest_zone(video_name: str, margin_px: int = 30):
    """
    Runs the yellow floor markings detector on the first frame of the video
    and suggests a candidate danger zone boundary.
    """
    # Check if first frame image is available
    first_frame_filename = f"{os.path.splitext(video_name)[0]}_first.jpg"
    first_frame_path = os.path.join("data/raw_videos", first_frame_filename)
    
    if not os.path.exists(first_frame_path):
        video_path = os.path.join("data/raw_videos", video_name)
        if os.path.exists(video_path):
            extract_first_frame(video_path, first_frame_path)
        else:
            raise HTTPException(status_code=404, detail="Video first frame not found, video file missing.")
            
    # Run yellow line detector (runs HSV thresholding + contour finding)
    suggested_coords = suggest_danger_zone(first_frame_path, margin_px)
    
    return {
        "video_name": video_name,
        "suggested_coordinates": suggested_coords  # Can be None if not found
    }

@app.post("/api/save-zone")
def save_zone(req: ZoneSaveRequest):
    """
    Saves danger zone coordinates to a JSON configuration file.
    """
    logger.info(f"Saving danger zone configuration for {req.video_name}: {req.zone_name}")
    try:
        config = save_zone_config(req.video_name, req.zone_name, req.coordinates)
        return {"status": "success", "config": config}
    except Exception as e:
        logger.error(f"Error saving zone config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")

@app.get("/api/load-zone/{video_name}")
def load_zone(video_name: str):
    """
    Loads saved danger zone coordinates for a given video.
    """
    config = load_zone_config(video_name)
    if not config:
        return JSONResponse(status_code=444, content={"message": "No danger zone saved for this video."})
    return config

# Real YOLO processing pipeline background task
async def real_processing_pipeline(video_name: str):
    logger.info(f"Starting real YOLO processing pipeline for {video_name}")
    processing_tasks[video_name] = {
        "status": "processing",
        "progress": 0,
        "output_video": None,
        "log_file": None,
        "error": None
    }
    
    video_path = os.path.join("data/raw_videos", video_name)
    config = load_zone_config(video_name)
    
    # Compute output filenames
    base_name = os.path.splitext(video_name)[0]
    output_video_filename = f"{base_name}_annotated.mp4"
    output_video_path = os.path.join("outputs", "annotated_videos", output_video_filename)
    
    output_log_filename = f"{base_name}_events.json"
    output_log_path = os.path.join("outputs", "event_logs", output_log_filename)
    
    def progress_callback(prog):
        processing_tasks[video_name]["progress"] = prog
        
    try:
        # Offload the heavy blocking OpenCV/YOLO track calculations to a thread pool executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            run_intrusion_pipeline,
            video_path,
            config,
            output_video_path,
            output_log_path,
            progress_callback
        )
        
        processing_tasks[video_name]["status"] = "completed"
        processing_tasks[video_name]["output_video"] = f"/outputs/annotated_videos/{output_video_filename}"
        processing_tasks[video_name]["log_file"] = f"/outputs/event_logs/{output_log_filename}"
        logger.info(f"Real YOLO processing completed for {video_name}")
        
    except Exception as e:
        logger.error(f"Real YOLO processing failed for {video_name}: {e}")
        processing_tasks[video_name]["status"] = "failed"
        processing_tasks[video_name]["error"] = str(e)

@app.post("/api/process-video")
def trigger_processing(video_name: str, background_tasks: BackgroundTasks):
    """
    Kicks off the YOLO processing pipeline on the video.
    """
    # Verify calibration exists
    config = load_zone_config(video_name)
    if not config:
        raise HTTPException(status_code=400, detail="Calibration configuration not found. Please calibrate first.")
        
    # Queue task (using real pipeline)
    background_tasks.add_task(real_processing_pipeline, video_name)
    return {"status": "started", "video_name": video_name}

@app.get("/api/status/{video_name}")
def get_task_status(video_name: str):
    """
    Returns the status and progress of the processing pipeline for the given video.
    """
    if video_name not in processing_tasks:
        # Check if configuration exists but process not run yet
        config = load_zone_config(video_name)
        if config:
            return {"status": "idle", "progress": 0, "message": "Calibrated but not processed yet."}
        return {"status": "uncalibrated", "progress": 0, "message": "Not calibrated yet."}
        
    return processing_tasks[video_name]
