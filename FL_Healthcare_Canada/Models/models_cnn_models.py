"""
Demo CNN architectures for medical imaging (PathMNIST)

Includes:
- SimpleCNN: Lightweight CNN for PathMNIST
- ResNet18PathMNIST: ResNet18 adapted for 28x28 images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class SimpleCNN(nn.Module):
    """
    Simple CNN for PathMNIST classification
    Lightweight architecture suitable for 28x28 images
    """
    
    def __init__(self, num_classes: int = 9, input_channels: int = 3, dropout: float = 0.5):
        """
        Initialize SimpleCNN
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB, 1 for grayscale)
            dropout: Dropout probability
        """
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1: 28x28 -> 14x14
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3: 7x7 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate flattened size: 128 channels * 3 * 3 = 1152
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class ResNet18PathMNIST(nn.Module):
    """
    ResNet18 adapted for PathMNIST (28x28 images)
    
    Modifications:
    - Smaller initial conv layer (3x3 instead of 7x7)
    - Removed initial max pooling
    - Adapted for small image size
    """
    
    def __init__(self, num_classes: int = 9, pretrained: bool = False):
        """
        Initialize ResNet18 for PathMNIST
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
        """
        super(ResNet18PathMNIST, self).__init__()
        
        # Load ResNet18
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Modify first conv layer for smaller images
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Modified: Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove max pooling (helps preserve spatial info in small images)
        self.resnet.maxpool = nn.Identity()
        
        # Modify final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, 28, 28)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        return self.resnet(x)


class CustomCNN(nn.Module):
    """
    Customizable CNN with configurable architecture
    """
    
    def __init__(
        self,
        num_classes: int = 9,
        input_channels: int = 3,
        conv_channels: list = [32, 64, 128],
        fc_dims: list = [256],
        dropout: float = 0.5,
        use_batchnorm: bool = True
    ):
        """
        Initialize custom CNN
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            conv_channels: List of channel sizes for conv layers
            fc_dims: List of hidden dimensions for FC layers
            dropout: Dropout probability
            use_batchnorm: Whether to use batch normalization
        """
        super(CustomCNN, self).__init__()
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for out_channels in conv_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate flattened size (assuming 28x28 input)
        # After len(conv_channels) pooling layers: 28 / (2^n)
        spatial_size = 28 // (2 ** len(conv_channels))
        flattened_size = conv_channels[-1] * spatial_size * spatial_size
        
        # Build fully connected layers
        fc_layers = []
        in_features = flattened_size
        
        for fc_dim in fc_dims:
            fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(nn.Linear(in_features, fc_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            in_features = fc_dim
        
        fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(in_features, num_classes))
        
        self.classifier = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_forward(model: nn.Module, input_shape: tuple = (1, 3, 28, 28)):
    """
    Test model forward pass
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, channels, height, width)
    """
    model.eval()
    x = torch.randn(*input_shape)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    print("="*60)
    print("Testing CNN Models for PathMNIST")
    print("="*60)
    
    # Test SimpleCNN
    print("\n1. SimpleCNN")
    print("-"*60)
    simple_cnn = SimpleCNN(num_classes=9)
    test_model_forward(simple_cnn)
    
    # Test ResNet18PathMNIST
    print("\n2. ResNet18PathMNIST")
    print("-"*60)
    resnet18 = ResNet18PathMNIST(num_classes=9, pretrained=False)
    test_model_forward(resnet18)
    
    # Test CustomCNN
    print("\n3. CustomCNN")
    print("-"*60)
    custom_cnn = CustomCNN(
        num_classes=9,
        conv_channels=[32, 64, 128],
        fc_dims=[256, 128],
        dropout=0.5
    )
    test_model_forward(custom_cnn)
    
    print("\n" + "="*60)
    print("All models tested successfully!")
    print("="*60)
