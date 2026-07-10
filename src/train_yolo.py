import os
import shutil
import yaml
import argparse
import random
import logging
from pathlib import Path
import torch
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def prepare_dataset(frames_dir: str, labels_dir: str, dataset_dir: str, split_ratio: float = 0.8):
    """
    Splits frames and labels into train/val subsets and organizes them in YOLO format.
    
    Structure:
    - dataset/
      - train/
        - images/
        - labels/
      - val/
        - images/
        - labels/
    """
    frames_path = Path(frames_dir)
    labels_path = Path(labels_dir)
    dataset_path = Path(dataset_dir)

    # Clear previous splits if they exist to start fresh
    if dataset_path.exists():
        logger.info(f"Cleaning previous dataset files in {dataset_path}...")
        import stat
        def remove_readonly(func, path, excinfo):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(dataset_path, onerror=remove_readonly)

    # Create directory structure
    for split in ["train", "val"]:
        (dataset_path / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # Gather all images across video directories
    image_label_pairs = []
    
    # Iterate over video folders inside frames
    video_dirs = [d for d in frames_path.iterdir() if d.is_dir()]
    for video_dir in video_dirs:
        video_name = video_dir.name
        video_labels_dir = labels_path / video_name
        
        if not video_labels_dir.exists():
            logger.warning(f"Label directory does not exist for video: {video_name}, skipping.")
            continue
            
        frame_files = [f for f in video_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        for img_file in frame_files:
            label_file = video_labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                image_label_pairs.append((img_file, label_file))

    if not image_label_pairs:
        raise ValueError(f"No matching image-label pairs found in {frames_dir} and {labels_dir}!")

    logger.info(f"Found {len(image_label_pairs)} image-label pairs. Splitting into train/val...")
    
    # Shuffle and split
    random.seed(42)  # For reproducible splits
    random.shuffle(image_label_pairs)
    
    split_index = int(len(image_label_pairs) * split_ratio)
    train_pairs = image_label_pairs[:split_index]
    val_pairs = image_label_pairs[split_index:]
    
    logger.info(f"Train split: {len(train_pairs)} images, Val split: {len(val_pairs)} images")

    # Helper function to copy files
    def copy_pairs(pairs, split_name):
        dest_img_dir = dataset_path / split_name / "images"
        dest_lbl_dir = dataset_path / split_name / "labels"
        
        for idx, (img_path, lbl_path) in enumerate(pairs):
            # To avoid name collisions across different videos, prefix with the video folder name
            prefix = img_path.parent.name + "_"
            dest_img_name = prefix + img_path.name
            dest_lbl_name = prefix + lbl_path.name
            
            shutil.copy(img_path, dest_img_dir / dest_img_name)
            shutil.copy(lbl_path, dest_lbl_dir / dest_lbl_name)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    logger.info("Dataset copies complete.")

    # Create the data.yaml file with absolute paths to avoid training path confusion
    data_config = {
        "path": str(dataset_path.resolve()),
        "train": os.path.join("train", "images"),
        "val": os.path.join("val", "images"),
        "nc": 4,
        "names": {
            0: "person",
            1: "crane_hook",
            2: "load_panel",
            3: "rubber_glove"
        }
    }
    
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
        
    logger.info(f"Generated YAML configuration at: {yaml_path}")
    return yaml_path

def train_yolo(data_yaml_path: Path, model_size: str, epochs: int, batch_size: int, models_dir: str):
    """
    Fine-tunes the YOLO model on the generated dataset and saves the weights.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing YOLO model: {model_size}...")
    model = YOLO(model_size)
    
    logger.info("Starting training...")
    # Sensible defaults for small dataset: data augmentation enabled, early stopping patience=10
    results = model.train(
        data=str(data_yaml_path.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        patience=10,
        project="runs/detect",
        name="crane_safety_train",
        augment=True,
        device=0 if torch.cuda.is_available() else "cpu",  # Auto-select GPU/CPU
        verbose=True
    )
    
    # Save the best model weights
    possible_paths = [
        Path("runs/detect/crane_safety_train/weights/best.pt"),
        Path("runs/detect/runs/detect/crane_safety_train/weights/best.pt")
    ]
    best_weights_path = None
    for p in possible_paths:
        if p.exists():
            best_weights_path = p
            break
            
    if best_weights_path is not None:
        final_weights_dest = models_path / "crane_safety_yolo.pt"
        shutil.copy(best_weights_path, final_weights_dest)
        logger.info(f"Best model weights saved to: {final_weights_dest}")
    else:
        raise FileNotFoundError(f"Could not find trained weights at any of the expected paths: {[str(p) for p in possible_paths]}")
        
    # Validate and log metrics
    logger.info("Running validation evaluations...")
    val_results = model.val()
    
    # Print metrics
    logger.info("YOLO validation results:")
    logger.info(f"mAP50: {val_results.results_dict['metrics/mAP50(B)']:.4f}")
    logger.info(f"mAP50-95: {val_results.results_dict['metrics/mAP50-95(B)']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and fine-tune YOLO model.")
    parser.add_argument("--frames_dir", type=str, default="data/extracted_frames", help="Path to extracted frames folder")
    parser.add_argument("--labels_dir", type=str, default="data/labels", help="Path to labels folder")
    parser.add_argument("--dataset_dir", type=str, default="data/dataset", help="Path to write train/val split dataset")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Pretrained YOLO base weight (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save final model weights")
    
    args = parser.parse_args()
    
    try:
        yaml_config_path = prepare_dataset(args.frames_dir, args.labels_dir, args.dataset_dir)
        train_yolo(yaml_config_path, args.model, args.epochs, args.batch, args.models_dir)
    except Exception as e:
        logger.exception("An error occurred during dataset prep or training:")
