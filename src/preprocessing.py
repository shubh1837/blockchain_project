import os
import numpy as np
import cv2
import pydicom
from PIL import Image

import torchvision.transforms as transforms

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
        img_np = dicom_data.pixel_array
        
        # Normalize to 0-255 if it's not uint8
        if img_np.dtype != np.uint8:
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255.0).astype(np.uint8)
            
        # Convert grayscale to RGB for DACNet
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
        img = Image.fromarray(img_np)
    else:
        # Load standard image using PIL and convert to RGB
        img = Image.open(file_path).convert('RGB')

    # Use PyTorch transforms to exactly match DACNet expectations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Process image (outputs shape C, H, W)
    img_tensor = transform(img)
    
    # Return as numpy array so the backend can unsqueeze it as expected
    return img_tensor.numpy()

if __name__ == "__main__":
    print("Preprocessing module ready. Create a dummy image to test.")
