import os
import numpy as np
import cv2
import pydicom
from PIL import Image

def load_and_preprocess_image(file_path, target_size=(224, 224)):
    """
    Loads an X-ray image (DICOM or standard formats) and preprocesses it.
    HIPAA compliance: In a production setting, this function ignores/strips PII 
    metadata from DICOM headers and only returns the pixel array.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.dcm':
        # Load DICOM
        dicom_data = pydicom.dcmread(file_path)
        # HIPAA compliance: Only extract the pixel data, ignore patient PII tags
        img = dicom_data.pixel_array
        
        # Normalize to 0-255 if it's not uint8
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)
    else:
        # Load standard image using OpenCV
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
             raise ValueError(f"Could not load image: {file_path}")

    # Resize image for the neural network
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Process for TorchXRayVision which expects inputs in the range [-1024, 1024]
    # For a general 0-255 image, we scale it.
    img_scaled = (img_resized.astype(np.float32) / 255.0) * 2048.0 - 1024.0
    
    # Add channel dimension (C, H, W) -> it expects 1 channel
    img_tensor = np.expand_dims(img_scaled, axis=0) 
    
    return img_tensor

if __name__ == "__main__":
    print("Preprocessing module ready. Create a dummy image to test.")
