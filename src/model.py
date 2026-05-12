import os
import torch
import torchxrayvision as xrv
import numpy as np
import cv2
import base64

class XRayModel:
    def __init__(self, model_name="densenet121-res224-all"):
        """
        Initializes the pre-trained TorchXRayVision model.
        """
        print(f"Loading TorchXRayVision model: {model_name}...")
        self.model = xrv.models.get_model(model_name)
        
        # Check for fine-tuned weights
        weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fine_tuned_weights.pth')
        if os.path.exists(weights_path):
            print(f"Found fine-tuned weights at {weights_path}. Loading them...")
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print("Fine-tuned weights loaded successfully.")
            
        # Ensure model and inputs are on the same device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.model.eval() # Set to evaluation mode
        self.pathologies = self.model.pathologies
        print(f"Model loaded successfully on {self.device}.")

    def predict(self, img_array, explain=True):
        """
        Run inference on the preprocessed image array and optionally generate an Explainable AI heatmap.
        """
        # Move input tensor to the same device as the model
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).to(self.device)
        
        if explain:
            img_tensor.requires_grad_()
            self.model.eval()
            outputs = self.model(img_tensor)
            
            # Format the output predictions
            results = {}
            for i, pathology in enumerate(self.pathologies):
                results[pathology] = float(outputs[0][i].cpu())
                
            sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
            
            # Explainability (Saliency Map for the top class)
            top_class = list(sorted_results.keys())[0]
            top_class_idx = self.pathologies.index(top_class)
            
            score = outputs[0][top_class_idx]
            self.model.zero_grad()
            score.backward()
            
            # Process gradients (move back to CPU for numpy operations)
            saliency = img_tensor.grad.data.cpu().abs().squeeze().numpy()
            saliency = saliency - saliency.min()
            saliency = saliency / (saliency.max() + 1e-8)
            saliency = np.uint8(255 * saliency)
            
            # Create a heatmap
            heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
            
            # Blend with original
            base_img = img_array[0] # assuming (1, H, W)
            base_img = base_img - base_img.min()
            base_img = base_img / (base_img.max() + 1e-8)
            base_img = np.uint8(255 * base_img)
            base_img_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
            
            blended = cv2.addWeighted(base_img_rgb, 0.5, heatmap, 0.5, 0)
            
            # Encode base64
            _, buffer = cv2.imencode('.png', blended)
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return sorted_results, heatmap_b64
            
        else:
            with torch.no_grad():
                outputs = self.model(img_tensor)
                
            results = {}
            for i, pathology in enumerate(self.pathologies):
                results[pathology] = float(outputs[0][i].cpu())
                
            sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
            return sorted_results, None

if __name__ == "__main__":
    print("Testing model initialization...")
    model = XRayModel()
    print("Pathologies supported:", model.pathologies)
