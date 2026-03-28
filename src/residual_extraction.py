import numpy as np
import warnings
import sys

def extract_residual(image: np.ndarray, wavelet='db8', levels=4):
    """
    Extracts sensor noise residual from a preprocessed image using Wavelet Denoising.
    Residual = Original - Wavelet-Denoised Image (Mihcak Filter style)
    
    Args:
        image: Numpy array of the preprocessed image (128, 128)
        wavelet: Type of wavelet to use (e.g., 'db8' Daubechies 8)
        levels: Number of wavelet decomposition levels
        
    Returns:
        Residual numpy array (128, 128)
    """
    try:
        from skimage.restoration import denoise_wavelet
    except ImportError:
        print("skimage not found! Ensure 'scikit-image' and 'PyWavelets' is installed.")
        raise
        
    # Ensure input is float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)
        
    # Apply Daubechies Wavelet Denoising (industry standard for PRNU extraction)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # We assume image is normalized to [0, 1]
        denoised_image = denoise_wavelet(image, method='BayesShrink', mode='soft', 
                                         wavelet_levels=levels, wavelet=wavelet, 
                                         rescale_sigma=True)
    
    # Subtraction to isolate the high-frequency sensor noise
    residual = image - denoised_image
    
    return residual
