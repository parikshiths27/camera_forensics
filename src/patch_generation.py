import numpy as np

def generate_patches(residual: np.ndarray, patch_size: int = 32):
    """
    Divides the residual image into non-overlapping patches.
    
    Args:
        residual: Numpy array of shape (128, 128)
        patch_size: Size of the square patch (default 32)
        
    Returns:
        Numpy array of shape (num_patches, patch_size, patch_size)
    """
    h, w = residual.shape
    
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image dimensions ({h}, {w}) must be divisible by patch size ({patch_size})")
        
    num_patches_y = h // patch_size
    num_patches_x = w // patch_size
    
    # Reshape the residual array into patches
    # (128, 128) -> (4, 32, 4, 32) -> (4, 4, 32, 32) -> (16, 32, 32)
    patches = (residual.reshape(num_patches_y, patch_size, num_patches_x, patch_size)
                       .swapaxes(1, 2)
                       .reshape(-1, patch_size, patch_size))
                       
    return patches

def process_pipeline(image_path: str):
    """
    Full pipeline to convert an image path into 16 residual patches.
    """
    from src.preprocessing import preprocess_image
    from src.residual_extraction import extract_residual
    
    # 1. Preprocess
    img = preprocess_image(image_path)
    # 2. Extract Residual
    res = extract_residual(img)
    # 3. Generate patches
    patches = generate_patches(res, patch_size=32)
    
    return patches
