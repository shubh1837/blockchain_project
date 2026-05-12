import torch
import torchxrayvision as xrv
import numpy as np

class XRayModel:
    def __init__(self, model_name="densenet121-res224-all"):
        """
        Initializes the pre-trained TorchXRayVision model.
        """
        print(f"Loading TorchXRayVision model: {model_name}...")
        self.model = xrv.models.get_model(model_name)
        self.model.eval() # Set to evaluation mode
        self.pathologies = self.model.pathologies
        print("Model loaded successfully.")

    def predict(self, img_array):
        """
        Run inference on the preprocessed image array.
        """
        # Convert numpy array to PyTorch tensor
        # Input shape should be (1, H, W) for a single grayscale image
        # Let's add batch dimension -> (1, 1, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        # Format the output predictions
        results = {}
        for i, pathology in enumerate(self.pathologies):
            results[pathology] = float(outputs[0][i])
            
        # Sort results by probability descending
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        return sorted_results

if __name__ == "__main__":
    print("Testing model initialization...")
    model = XRayModel()
    print("Pathologies supported:", model.pathologies)
