import rasterio
import cv2
import numpy as np

def bicubic_interpolation_opencv(image, scale_factor=10):
    return cv2.resize(
        image, 
        None, 
        fx=scale_factor, 
        fy=scale_factor, 
        interpolation=cv2.INTER_CUBIC
    )

def extract_elev_Tiff(filepath_tiff):
    with rasterio.open(filepath_tiff) as src:
        return src.read(1).astype(np.float64)  # Já converte para float