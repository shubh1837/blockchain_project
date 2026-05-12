import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNet121(nn.Module):
    def __init__(self, classCount=14, isTrained=False):
        super(DenseNet121, self).__init__()
        
        # Load the base DenseNet121 from torchvision
        weights = DenseNet121_Weights.IMAGENET1K_V1 if isTrained else None
        self.densenet121 = densenet121(weights=weights)
        
        # Get the number of input features for the classifier
        kernelCount = self.densenet121.classifier.in_features
        
        # Replace the classifier with just a Linear layer (no Sigmoid)
        # DACNet uses FocalLoss with logits, so Sigmoid is applied during inference externally
        self.densenet121.classifier = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.densenet121(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
