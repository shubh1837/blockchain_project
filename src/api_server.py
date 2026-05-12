import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import pydicom

from src.model import XRayModel
from src.preprocessing import load_and_preprocess_image
from src.ssh_security import SSHGenerator
from src.storage import IPFSStorage
from src.blockchain_client import BlockchainClient
import sys

node_port = "8000"
for i, arg in enumerate(sys.argv):
    if arg == "--port" and i + 1 < len(sys.argv):
        node_port = sys.argv[i+1]

def log_message(msg):
    print(f"[NODE PORT {node_port}] {msg}")

app = FastAPI(title="DacNet Diagnostic API with Persistent History")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log_message("Mounting PyTorch Ecosystem for API...")
model_wrapper = XRayModel()
optimizer = optim.Adam(model_wrapper.model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_wrapper.model.to(device)

storage = IPFSStorage()
ssh_gen = SSHGenerator()
bc = BlockchainClient()

HISTORY_DIR = f"data/node_{node_port}_history"
TRAINED_DIR = os.path.join(HISTORY_DIR, "trained")
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(TRAINED_DIR, exist_ok=True)

class FeedbackData(BaseModel):
    job_id: str
    confirmed_pathologies: dict

@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    if not ext: ext = ".png"
    history_path = os.path.join(HISTORY_DIR, f"{job_id}{ext}")
    
    try:
        with open(history_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        log_message(f"[{job_id}] Request Received: Secured to history vault.")
        preprocessed_img = load_and_preprocess_image(history_path)
        
        predictions, heatmap_b64 = model_wrapper.predict(preprocessed_img, explain=True)
        formatted_preds = [{"name": k, "score": v} for k, v in predictions.items()]
        
        ssh_hash = ssh_gen.generate_ssh(history_path)
        cid = storage.upload_to_ipfs(history_path)
        tx_id = bc.log_diagnostic_record(cid, ssh_hash)
        
        return {
            "status": "success",
            "job_id": job_id,
            "filename": file.filename,
            "pathologies": formatted_preds,
            "heatmap_base64": heatmap_b64,
            "ssh_hash": ssh_hash,
            "cid": cid,
            "tx_id": tx_id,
            "message": "Inference Complete. File banked securely for delayed feedback."
        }
    except Exception as e:
        log_message(f"[{job_id}] Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/train")
async def active_learning_step(feedback: FeedbackData):
    log_message(f"[{feedback.job_id}] INITIATING DELAYED ACTIVE LEARNING")
    
    # Locate the historically saved file
    found_path = None
    for file in os.listdir(HISTORY_DIR):
        if file.startswith(feedback.job_id):
            found_path = os.path.join(HISTORY_DIR, file)
            break
            
    if not found_path or os.path.isdir(found_path):
         return {"status": "error", "message": "Original image not found in history vault."}
         
    try:
        preprocessed_img = load_and_preprocess_image(found_path)
        img_tensor = torch.from_numpy(preprocessed_img).unsqueeze(0).to(device)
        
        ground_truth = []
        for path in model_wrapper.pathologies:
             score = feedback.confirmed_pathologies.get(path, 0.0)
             ground_truth.append(score)
             
        target_tensor = torch.tensor([ground_truth]).to(device).float()
        
        model_wrapper.model.train()
        optimizer.zero_grad()
        
        outputs = model_wrapper.model(img_tensor)
        loss = criterion(outputs, target_tensor)
        loss.backward()
        optimizer.step()
        
        log_message(f"[{feedback.job_id}] Weights adjusted recursively. Local Loss: {loss.item():.4f}")
        
        # Archiving file from pending to trained zone
        shutil.move(found_path, os.path.join(TRAINED_DIR, os.path.basename(found_path)))
        
        return {
            "status": "success", 
            "loss": float(loss.item()),
            "message": "Weights updated correctly based on delayed ground truth."
        }
    except Exception as e:
        log_message(f"[{feedback.job_id}] Training Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    log_message("API Persistent Server Active.")
