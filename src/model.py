import os
import torch
import numpy as np
import cv2
import base64
import re
from src.dacnet import DenseNet121

class XRayModel:
    def __init__(self):
        """
        Initializes the pre-trained DACNet model.
        """
        print("Loading DACNet model...")
        self.model = DenseNet121(classCount=14, isTrained=False)
        
        # Define the exact 14 pathologies trained on CheXNet
        self.pathologies = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        # Load weights
        weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dacnet.pth')
        if os.path.exists(weights_path):
            print(f"Found DACNet weights at {weights_path}. Loading them...")
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            
            # The DACNet weights were saved directly from a torchvision densenet121.
            # So the keys match the internal self.model.densenet121.
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Remove any 'module.' just in case
            clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.model.densenet121.load_state_dict(clean_state_dict)
            print("DACNet weights loaded successfully.")
        else:
            print(f"WARNING: DACNet weights not found at {weights_path}")
            
        # Ensure model and inputs are on the same device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.model.eval() # Set to evaluation mode
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
            
            # DACNet outputs logits, so we apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Format the output predictions
            results = {}
            for i, pathology in enumerate(self.pathologies):
                results[pathology] = float(probs[0][i].cpu())
                
            sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
            
            # Explainability (Saliency Map for the top class)
            top_class = list(sorted_results.keys())[0]
            top_class_idx = self.pathologies.index(top_class)
            
            score = outputs[0][top_class_idx]
            self.model.zero_grad()
            score.backward()
            
            # Process gradients (move back to CPU for numpy operations)
            # Saliency is (1, 3, 224, 224). We squeeze to get (3, 224, 224) and max across channels to get (224, 224).
            saliency = img_tensor.grad.data.cpu().abs().squeeze().numpy()
            saliency = np.max(saliency, axis=0)
            
            saliency = saliency - saliency.min()
            saliency = saliency / (saliency.max() + 1e-8)
            saliency = np.uint8(255 * saliency)
            
            # Create a heatmap
            heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
            
            # Blend with original
            # img_array is (3, H, W), we need (H, W, 3) for cv2 operations
            base_img = np.transpose(img_array, (1, 2, 0))
            base_img = base_img - base_img.min()
            base_img = base_img / (base_img.max() + 1e-8)
            base_img_rgb = np.uint8(255 * base_img)
            
            # Convert RGB (from torchvision transforms) to BGR for OpenCV blending
            base_img_bgr = cv2.cvtColor(base_img_rgb, cv2.COLOR_RGB2BGR)
            
            blended = cv2.addWeighted(base_img_bgr, 0.5, heatmap, 0.5, 0)
            
            # Encode base64
            _, buffer = cv2.imencode('.png', blended)
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return sorted_results, heatmap_b64
            
        else:
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.sigmoid(outputs)
                
            results = {}
            for i, pathology in enumerate(self.pathologies):
                results[pathology] = float(probs[0][i].cpu())
                
            sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
            return sorted_results, None

if __name__ == "__main__":
    print("Testing model initialization...")
    model = XRayModel()
    print("Pathologies supported:", model.pathologies)
