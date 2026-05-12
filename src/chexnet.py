import torch
import torch.nn as nn
import torchvision

class DenseNet121(nn.Module):
    def __init__(self, classCount=14, isTrained=False):
        super(DenseNet121, self).__init__()
        
        # Load the base DenseNet121 from torchvision
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        
        # Get the number of input features for the classifier
        kernelCount = self.densenet121.classifier.in_features
        
        # Replace the classifier with a Linear layer followed by a Sigmoid
        # The Sigmoid ensures outputs are probabilities between 0 and 1
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(kernelCount, classCount), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
