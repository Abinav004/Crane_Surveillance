import os
import json
import logging
from datetime import datetime

logger = logging.getLogger("zone_config")
CONFIG_DIR = os.path.join("config", "danger_zones")

def get_config_path(video_name: str) -> str:
    """
    Returns the file path for the danger zone configuration of a video.
    """
    base_name = os.path.splitext(video_name)[0]
    return os.path.join(CONFIG_DIR, f"{base_name}_zones.json")

def save_zone_config(video_name: str, zone_name: str, coordinates: list) -> dict:
    """
    Saves the zone name and coordinate details for a video as a JSON file.
    """
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config_path = get_config_path(video_name)
    
    config_data = {
        "zone_name": zone_name,
        "video_name": video_name,
        "coordinates": coordinates,  # Expected format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        "created_timestamp": datetime.now().isoformat()
    }
    
    try:
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Successfully saved zone config to {config_path}")
        return config_data
    except Exception as e:
        logger.error(f"Failed to save zone config to {config_path}: {e}")
        raise e

def load_zone_config(video_name: str) -> dict:
    """
    Loads the zone config for a given video. Returns None if it doesn't exist.
    """
    config_path = get_config_path(video_name)
    if not os.path.exists(config_path):
        logger.warning(f"No config file found at {config_path}")
        return None
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded zone config from {config_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load zone config from {config_path}: {e}")
        return None
