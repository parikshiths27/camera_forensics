import os
import shutil
import random
from pathlib import Path

def split_dataset(raw_dir: str, output_dir: str, train_ratio: float = 0.8):
    """
    Splits the raw dataset into train and test sets.
    Expected raw_dir structure:
    raw_dir/
      class_1/
        img1.jpg
      class_2/
        img2.jpg
    """
    raw_path = Path(raw_dir)
    train_path = Path(output_dir) / 'train'
    test_path = Path(output_dir) / 'test'

    if not raw_path.exists():
        print(f"Error: Raw dataset path '{raw_dir}' does not exist.")
        return

    for class_dir in raw_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        
        # Create output directories for this class
        (train_path / class_name).mkdir(parents=True, exist_ok=True)
        (test_path / class_name).mkdir(parents=True, exist_ok=True)
        
        # Get all images
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        print(f"Class '{class_name}': {len(train_images)} train, {len(test_images)} test")
        
        for img in train_images:
            shutil.copy2(img, train_path / class_name / img.name)
            
        for img in test_images:
            shutil.copy2(img, test_path / class_name / img.name)
            
    print("Dataset splitting complete.")

if __name__ == "__main__":
    # Default paths
    RAW_DIR = r"c:\Projects\camera_forensics\dataset_raw"
    DATASET_DIR = r"c:\Projects\camera_forensics\dataset"
    
    print(f"Splitting dataset from {RAW_DIR} into {DATASET_DIR}")
    split_dataset(RAW_DIR, DATASET_DIR)
