import flwr as fl
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchxrayvision as xrv
import os
import numpy as np

class XRayHospitalClient(fl.client.NumPyClient):
    def __init__(self, data_dir="data/NIH", limit_memory=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Hospital Node AI initialised on {self.device}.")
        
        # 1. Initialize TorchXRayVision Model Architecture
        # Using the NIH-specific backbone ensures the output layer precisely matches 
        # the 14 pathologies of the NIH dataset to prevent dimension errors.
        self.model = xrv.models.get_model("densenet121-res224-nih")
        self.model.to(self.device)
        
        # 2. Setup PyTorch Loss and Optimizer
        # BCEWithLogitsLoss is required for multi-label binary tasks
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 3. Load Actual Data
        csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
        img_dir = os.path.join(data_dir, "images")
        
        if not os.path.exists(csv_path):
             raise FileNotFoundError("NIH Dataset not found. Please run scripts/setup_nih_dataset.py first.")
             
        transform = xrv.datasets.XRayCenterCrop()
        self.dataset = xrv.datasets.NIH_Dataset(
            imgpath=img_dir,
            csvpath=csv_path,
            transform=transform
        )
        
        # Store dataset pathology map before any Subsetting happens
        self.dataset_pathologies = self.dataset.pathologies

        # Device Limitation: Subset the dataset drastically to prevent OOM
        # 42GB dataset / 112k images will crash most home PCs if loaded in memory
        if limit_memory:
             # Train only on the first 40 examples
             print("Applying Device Limitation: Downsampling dataset representation to 40 samples.")
             indices = list(range(min(40, len(self.dataset))))
             self.dataset = Subset(self.dataset, indices)
        
        # Split into Train/Test locally
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        
        # DataLoader handles batching
        self.train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        self.val_loader = DataLoader(test_ds, batch_size=8)
        
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
             
             # Align the 18 default model outputs to the 14 dataset labels
             path_indices = [self.model.pathologies.index(path) for path in self.dataset_pathologies]
             outputs_aligned = outputs[:, path_indices]
             
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
                 
                 path_indices = [self.model.pathologies.index(path) for path in self.dataset_pathologies]
                 outputs_aligned = outputs[:, path_indices]
                 
                 loss = self.criterion(outputs_aligned, labels)
                 val_loss += loss.item()
                 
        final_loss = val_loss / max(1, len(self.val_loader))
        return float(final_loss), len(self.val_loader.dataset), {"loss": final_loss}

if __name__ == "__main__":
    print("Starting Deep Learning Client Node...")
    client = XRayHospitalClient(limit_memory=True)
    
    # We execute a dry run of standard PyTorch locally for proof of functionality.
    # In a real FL system, we pass control to the Flower client:
    # fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    
    print("\n--- Starting Standalone Test Training Loop ---")
    client.fit(parameters=client.get_parameters(config={}), config={})
    print("\n--- Executing Validation Loop ---")
    loss, count, d = client.evaluate(parameters=client.get_parameters(config={}), config={})
    print(f"Validation Loss: {loss:.4f} across {count} samples.")
