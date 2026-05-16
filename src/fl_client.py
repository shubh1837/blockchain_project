import flwr as fl
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchxrayvision as xrv
import os
import json
import time
import numpy as np
import socket
from src.model import XRayModel
from src.preprocessing import load_and_preprocess_image

def discover_server(port=55555):
    """Listens for the UDP beacon from the FL server."""
    print("Listening for Auto-Discovery Beacon from Central Server...")
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Enable SO_REUSEADDR and SO_BROADCAST to ensure we can bind to it
    client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client.bind(('', port))
    
    while True:
        data, addr = client.recvfrom(1024)
        msg = data.decode('utf-8')
        if msg.startswith("DACNET_FL_SERVER:"):
            fl_port = msg.split(":")[1]
            server_ip = addr[0]
            print(f"Auto-Discovery Success! Found Central Server at {server_ip}:{fl_port}")
            client.close()
            return f"{server_ip}:{fl_port}"


class CustomHistoryDataset(torch.utils.data.Dataset):
    def __init__(self, history_dir):
        self.history_dir = history_dir
        self.samples = []
        
        if os.path.exists(history_dir):
            for file in os.listdir(history_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(history_dir, file)
                    json_path = os.path.splitext(img_path)[0] + ".json"
                    if os.path.exists(json_path):
                        self.samples.append((img_path, json_path))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        
        # Use exact preprocessing as API to match DACNet (Outputs C,H,W numpy)
        img_array = load_and_preprocess_image(img_path)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            labels = np.array(data['labels'], dtype=np.float32)
            
        return {"img": torch.from_numpy(img_array), "lab": torch.from_numpy(labels)}
class XRayHospitalClient(fl.client.NumPyClient):
    def __init__(self, node_port="8000"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Hospital Node AI initialised on {self.device}.")
        
        # 1. Initialize DACNet directly to match backend
        model_wrapper = XRayModel()
        self.model = model_wrapper.model
        self.model.to(self.device)
        self.dataset_pathologies = model_wrapper.pathologies
        
        # 2. Setup PyTorch Loss and Optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 3. Wait for Local Hospital Data (Prevention Method)
        history_dir = f"data/node_{node_port}_history/trained"
        print(f"Scanning for doctor-confirmed images in {history_dir}...")
        
        os.makedirs(history_dir, exist_ok=True)
        
        while True:
            self.dataset = CustomHistoryDataset(history_dir)
            if len(self.dataset) > 0:
                print(f"Found {len(self.dataset)} confirmed images! Proceeding to join Federated network.")
                break
            print("Dataset empty. Please upload and confirm at least 1 image on the dashboard. Waiting 5s...")
            time.sleep(5)
            
        # Split into Train/Test locally
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        
        # If very few images, put everything in train
        if test_size == 0:
            train_ds = self.dataset
            test_ds = self.dataset
        else:
            train_ds, test_ds = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        
        # DataLoader handles batching
        self.train_loader = DataLoader(train_ds, batch_size=min(8, len(train_ds)), shuffle=True)
        self.val_loader = DataLoader(test_ds, batch_size=min(8, len(test_ds)))
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Authentic PyTorch Training Loop."""
        self.set_parameters(parameters)
        print("Training model locally on private data...")
        self.model.train()
        
        # 1 Epoch of training step
        epoch_loss = 0.0
        for batch in self.train_loader:
             # xrv datasets return dict {"img": tensor(B, 1, H, W), "lab": tensor(B, C)}
             images = batch["img"].to(self.device).float()
             labels = batch["lab"].to(self.device).float()
             
             self.optimizer.zero_grad()
             outputs = self.model(images)
             
             # The dataset already provides aligned labels because we created them directly from XRayModel.pathologies!
             outputs_aligned = outputs
             
             loss = self.criterion(outputs_aligned, labels)
             loss.backward()
             self.optimizer.step()
             epoch_loss += loss.item()
             
        print(f"Local Epoch Completed - Loss: {epoch_loss/len(self.train_loader):.4f}")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Authentic Evaluation on Test subset."""
        self.set_parameters(parameters)
        print("Evaluating global model against local validation set...")
        self.model.eval()
        
        val_loss = 0.0
        with torch.no_grad():
             for batch in self.val_loader:
                 images = batch["img"].to(self.device).float()
                 labels = batch["lab"].to(self.device).float()
                 outputs = self.model(images)
                 
                 # Labels are already perfectly aligned to XRayModel
                 outputs_aligned = outputs
                 
                 loss = self.criterion(outputs_aligned, labels)
                 val_loss += loss.item()
                 
        final_loss = val_loss / max(1, len(self.val_loader))
        return float(final_loss), len(self.val_loader.dataset), {"loss": final_loss}

if __name__ == "__main__":
    import sys
    
    server_ip = "auto"
    is_standalone = False
    node_port = "8000"
    
    for i, arg in enumerate(sys.argv):
        if arg == "--server" and i + 1 < len(sys.argv):
            server_ip = sys.argv[i+1]
        if arg == "--port" and i + 1 < len(sys.argv):
            node_port = sys.argv[i+1]
        if arg == "--standalone":
            is_standalone = True

    print(f"Starting Deep Learning Client Node (Port {node_port})...")
    client = XRayHospitalClient(node_port=node_port)
    
    if is_standalone:
        print("\n--- Starting Standalone Test Training Loop ---")
        client.fit(parameters=client.get_parameters(config={}), config={})
        print("\n--- Executing Validation Loop ---")
        loss, count, d = client.evaluate(parameters=client.get_parameters(config={}), config={})
        print(f"Validation Loss: {loss:.4f} across {count} samples.")
    else:
        if server_ip == "auto":
            server_ip = discover_server()
            
        print(f"\n--- Connecting to Federated Server at {server_ip} ---")
        fl.client.start_numpy_client(server_address=server_ip, client=client)

