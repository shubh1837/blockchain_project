import os
import pandas as pd
import numpy as np
import cv2

DATA_DIR = "data/NIH"
IMG_DIR = os.path.join(DATA_DIR, "images")
csv_path = os.path.join(DATA_DIR, "Data_Entry_2017.csv")

def create_synthetic_nih_subset(num_samples=50):
    """
    Attempts to download the official ~2GB Kaggle sample dataset if credentials exist.
    Otherwise, generates a structural mock.
    """
    kaggle_creds = os.path.expanduser("~/.kaggle/kaggle.json")
    
    if os.path.exists(kaggle_creds):
        print("Detected kaggle.json! Attempting to download official NIH Sample Dataset...")
        try:
            import kaggle
            print("Downloading 'nih-chest-xrays/sample' from Kaggle...")
            # Automatically unzips into data/NIH
            kaggle.api.dataset_download_cli("nih-chest-xrays/sample", path=DATA_DIR, unzip=True)
            print("Successfully downloaded Official Kaggle Sample!")
            return
        except Exception as e:
            print(f"Kaggle API download failed: {e}. Falling back to MOCK generation.")
            
    os.makedirs(IMG_DIR, exist_ok=True)
    print(f"Generating {num_samples} bounded NIH-style synthetic X-Rays for local training...")
    
    # Standard 14 pathologies in Kaggle NIH dataset
    pathologies = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
    ]
    
    records = []
    
    for i in range(num_samples):
        img_name = f"000000{i:02d}_000.png"
        img_path = os.path.join(IMG_DIR, img_name)
        
        # Create a visually "noisy" dummy X-Ray
        dummy_img = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
        cv2.imwrite(img_path, dummy_img)
        
        # Assign a random pathology (or 'No Finding')
        if np.random.rand() > 0.5:
             label = "No Finding"
        else:
             num_labels = np.random.randint(1, 3)
             label = "|".join(np.random.choice(pathologies[:-1], num_labels, replace=False))
             
        # Create mock record strictly matching Kaggle's Data_Entry_2017.csv structure
        records.append({
            "Image Index": img_name,
            "Finding Labels": label,
            "Follow-up #": 0,
            "Patient ID": i,
            "Patient Age": int(np.random.randint(20, 80)),
            "Patient Gender": np.random.choice(["M", "F"]),
            "View Position": np.random.choice(["PA", "AP"]),
            "OriginalImage[Width": 1024,
            "Height]": 1024,
            "OriginalImagePixelSpacing[x": 0.143,
            "y]": 0.143
        })
        
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated NIH Dataset MOCK at {csv_path}")
    print("Ready for Memory-Limited Training!")

if __name__ == "__main__":
    if not os.path.exists(csv_path):
        create_synthetic_nih_subset(50)
    else:
        print("NIH Dataset (or mock) already exists at ./data/NIH")
