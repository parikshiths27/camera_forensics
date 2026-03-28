import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CameraForensicsCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CameraForensicsCNN, self).__init__()
        
        # Load a pre-trained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the first convolutional layer to accept 1-channel grayscale patches
        # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # ResNet18 uses 512 features before the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        
        # Replace the final fully connected layer for our 3 classes, adding dropout for robustness
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
