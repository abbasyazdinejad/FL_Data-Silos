"""
Demo Federated learning model

"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import copy


class TabularNN(nn.Module):
    """
    Feedforward neural network 
    Used for cancer detection and other tabular healthcare data
    """
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 2,
        dropout: float = 0.3,
        use_batchnorm: bool = True
    ):
        """
       
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout: Dropout probability
            use_batchnorm: Whether to use batch normalization
        """
        super(TabularNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Output logits (batch_size, output_dim)
        """
        return self.network(x)


class MultiTaskNN(nn.Module):
    """
    Multi-task neural network for tabular data
    Can predict multiple targets simultaneously
    """
    
    def __init__(
        self,
        input_dim: int,
        shared_hidden_dims: List[int] = [128, 64],
        task_hidden_dims: List[int] = [32],
        output_dims: List[int] = [2, 3],
        dropout: float = 0.3
    ):
        """
        Initialize multi-task network
        
        Args:
            input_dim: Input feature dimension
            shared_hidden_dims: Shared layer dimensions
            task_hidden_dims: Task-specific layer dimensions
            output_dims: List of output dimensions for each task
            dropout: Dropout probability
        """
        super(MultiTaskNN, self).__init__()
        
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.task_heads = nn.ModuleList()
        for output_dim in output_dims:
            task_layers = []
            task_prev_dim = prev_dim
            
            for task_hidden_dim in task_hidden_dims:
                task_layers.extend([
                    nn.Linear(task_prev_dim, task_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                task_prev_dim = task_hidden_dim
            
            task_layers.append(nn.Linear(task_prev_dim, output_dim))
            self.task_heads.append(nn.Sequential(*task_layers))
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            List of outputs for each task
        """
        shared_features = self.shared_network(x)
        outputs = [head(shared_features) for head in self.task_heads]
        return outputs


class FederatedModel:
    """

    Provides utilities for parameter management and model operations
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize federated model wrapper
        
        Args:
            model: PyTorch model
            device: Computing device
        """
        self.model = model.to(device)
        self.device = device
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get model parameters as dictionary
        
        Returns:
            Dict mapping parameter names to tensors
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set model parameters from dictionary
        
        Args:
            parameters: Dict of parameter tensors
        """
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data = parameters[name].to(self.device).clone()
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Get model gradients as dictionary
        
        Returns:
            Dict mapping parameter names to gradient tensors
        """
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.data.clone()
        return gradients
    
    def get_state_dict(self):
        """Get model state dict"""
        return copy.deepcopy(self.model.state_dict())
    
    def load_state_dict(self, state_dict):
        """Load model state dict"""
        self.model.load_state_dict(state_dict)
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def train(self):
        """Set model to training mode"""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
    
    def save(self, path: str):
        """
        Save model to file
        
        Args:
            path: File path
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__
        }, path)
    
    @classmethod
    def load(cls, path: str, model: nn.Module, device: torch.device):
        """
        Load model 
        
        Args:
            path: File path
            model: Model instance (architecture)
            device: Computing device
            
        Returns:
            FederatedModel instance
        """
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return cls(model, device)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class LogisticRegression(nn.Module):
    """
    Logistic regression for binary classification
    Useful for baseline comparisons
    """
    
    def __init__(self, input_dim: int, num_classes: int = 2):
        """
        Initialize logistic regression
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes (2 for binary)
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


def create_model_for_task(
    task_name: str,
    input_dim: int,
    num_classes: int,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3
) -> nn.Module:
    """
    Factory function to create models for different tasks
    
    Args:
        task_name: Name of the task ('cancer', 'pandemic', 'rare_disease', 'imaging')
        input_dim: Input dimension
        num_classes: Number of output classes
        hidden_dims: Hidden layer dimensions (optional)
        dropout: Dropout probability
        
    Returns:
        PyTorch model
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]
    
    if task_name in ['cancer', 'pandemic', 'rare_disease']:
        return TabularNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            dropout=dropout
        )
    elif task_name == 'imaging':
        from .cnn_models import ResNet18PathMNIST
        return ResNet18PathMNIST(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown task: {task_name}")


if __name__ == "__main__":
    print("="*60)
    print("Testing Federated Models")
    print("="*60)
    
    # Test TabularNN
    print("\n1. TabularNN for Cancer Detection")
    print("-"*60)
    model = TabularNN(
        input_dim=10,
        hidden_dims=[64, 32],
        output_dim=2,
        dropout=0.3
    )
    
    # Test forward pass
    x = torch.randn(32, 10)  # Batch of 32 samples, 10 features
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test FederatedModel wrapper
    print("\n2. FederatedModel Wrapper")
    print("-"*60)
    device = torch.device('cpu')
    fed_model = FederatedModel(model, device)
    
    # Get parameters
    params = fed_model.get_parameters()
    print(f"Number of parameter tensors: {len(params)}")
    
    # Test parameter setting
    new_params = {name: torch.randn_like(param) for name, param in params.items()}
    fed_model.set_parameters(new_params)
    print("Parameters updated successfully")
    
    # Test LogisticRegression
    print("\n3. Logistic Regression (Baseline)")
    print("-"*60)
    lr_model = LogisticRegression(input_dim=10, num_classes=2)
    output = lr_model(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in lr_model.parameters()):,}")
    
    # Test MultiTaskNN
    print("\n4. Multi-Task Neural Network")
    print("-"*60)
    mt_model = MultiTaskNN(
        input_dim=10,
        shared_hidden_dims=[128, 64],
        task_hidden_dims=[32],
        output_dims=[2, 3],  # Task 1: binary, Task 2: 3-class
        dropout=0.3
    )
    outputs = mt_model(x)
    print(f"Task 1 output shape: {outputs[0].shape}")
    print(f"Task 2 output shape: {outputs[1].shape}")
    
    print("\n" + "="*60)
    print("All models tested successfully!")
    print("="*60)
