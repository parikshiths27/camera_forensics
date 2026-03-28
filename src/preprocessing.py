import cv2
import numpy as np

def preprocess_image(image_path: str):
    """
    Reads an image from path and performs preprocessing:
    1. Grayscale conversion (removes color bias)
    2. Resize to 128x128 (uniform input size)
    3. Normalization (focuses on intensity)
    
    Returns a normalized grayscale image (128, 128) as a numpy array.
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")
        
    # CRITICAL FORENSIC FIX: Do NOT resize! Resizing blends pixels and destroys PRNU sensor noise.
    # We must crop a 128x128 region from the original image to preserve the true 1:1 pixel hardware noise.
    h, w = img.shape
    crop_size = 128
    if h >= crop_size and w >= crop_size:
        start_y = h // 2 - crop_size // 2
        start_x = w // 2 - crop_size // 2
        img_processed = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
    else:
        # Fallback only if the image is astronomically small
        img_processed = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    
    # Normalize to 0-1
    img_normalized = img_processed.astype(np.float32) / 255.0
    
    return img_normalized
