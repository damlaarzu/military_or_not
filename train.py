#!/usr/bin/env python3
"""
YOLOv11l civilian and military vehicle detection training
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import warnings

DATA_YAML_PATH = "dataset.yaml"
OUTPUT_DIR = Path("datasets/merged")


warnings.filterwarnings("ignore")

def setup_training_config():
    """Sets up the configuration parameters optimized for training."""
    
    training_config = {
        'model': 'yolo11l.pt', 
        'data': DATA_YAML_PATH, 
    
        # Train parameters
        'epochs': 300,
        'patience': 25, # if not improved for 25 epochs
        'batch': 16,     
        'imgsz': 640,   
        'save': True,
        'save_period': 25, # Each 25 epochs saves the model
        'cache': 'disk',   # Caches training data to disk
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 4, # Uses as many workers as CPU cores
        'project': 'runs/train',
        'name': 'military_or_not',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        
        # Optimizer and LR scheduler
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        # Data augmentation
        'degrees': 10.0,
        'scale': 0.5,
        'shear': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,

        # Confidence and NMS settings
        'conf': 0.25, # Bounding box confidence threshold
        'iou': 0.7,   # Non-Maximum Suppression (NMS) IoU threshold

        # Other settings
        'val': True, # Evaluate on validation set
        'plots': True, # Generate metrics and plots
    }
    
    return training_config

def validate_dataset_structure():
    """Validates the dataset structure and files."""
    
    print("Validating dataset structure...")
    
    # Check images and labels folders
    images_train_path = OUTPUT_DIR / "images" / "train"
    labels_train_path = OUTPUT_DIR / "labels" / "train"
    
    if not images_train_path.exists() or not labels_train_path.exists():
        raise FileNotFoundError(
            f"Training folders not found. Please ensure you have run the `robocopy` commands and the `convert_nwpu.py` script correctly."
        )

    # Count files
    img_count = len(list(images_train_path.glob("*.jpg")))
    label_count = len(list(labels_train_path.glob("*.txt")))

    print(f"✓ Training set contains {img_count} images and {label_count} labels.")

    # Check data.yaml file
    yaml_path = Path(DATA_YAML_PATH)
    if not yaml_path.exists():
        raise FileNotFoundError(f"'{DATA_YAML_PATH}' file not found. Please create this file.")

    print(f"✓ '{DATA_YAML_PATH}' file found.")

    return True

def main():
    """Main training function"""

    print("Starting training with YOLOv11l for Civilian/Military Vehicle Detection")
    print("=" * 60)

    # Validate dataset
    try:
        validate_dataset_structure()
        print(" dataset validation successful!\n")
    except Exception as e:
        print(f" dataset validation failed: {e}")
        return

    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}\n")

    # Get training configuration
    config = setup_training_config()

    print(" Training Configuration:")
    print(f"  - Model: {config['model']}")
    print(f"  - Optimizer: {config['optimizer']}")
    print(f"  - Epochs: {config['epochs']}")
    print(f"  - Batch Size: {config['batch']}")
    print(f"  - Image Size: {config['imgsz']}")
    print("")
    
    try:
        # Start YOLO model
        print(" YOLOv11 model loading...")
        model = YOLO(config['model'])

        # Start training
        print("This process may take hours depending on your hardware.")
        print("=" * 60)
        
        results = model.train(
            **{k: v for k, v in config.items() if k != 'model'}
        )
        
        print("\n" + "=" * 60)
        print(" Training completed successfully!")
        print(f" Results saved to: {results.save_dir}")
        print(" Check 'runs/train' folder for detailed results.")

    except Exception as e:
        print(f"\n Training failed with error: {e}")
        print("Please check your dataset structure and 'dataset.yaml' file.")
        return

    print("\n Training script completed!")

if __name__ == "__main__":
    main()