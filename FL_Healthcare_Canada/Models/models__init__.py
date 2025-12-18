"""
Demo Model architectures for Federated Learning Healthcare

Includes:
- CNN models for medical imaging (PathMNIST)
- Tabular models for cancer detection and other healthcare tasks
- Federated learning model wrappers
"""

# CNN Models for Medical Imaging
from .cnn_models import (
    SimpleCNN,
    ResNet18PathMNIST,
    CustomCNN,
    count_parameters,
    test_model_forward
)

# Tabular and Federated Models
from .federated_models import (
    TabularNN,
    MultiTaskNN,
    FederatedModel,
    LogisticRegression,
    create_model_for_task
)

__all__ = [
    # CNN Models
    'SimpleCNN',
    'ResNet18PathMNIST',
    'CustomCNN',
    'count_parameters',
    'test_model_forward',
    
    # Tabular Models
    'TabularNN',
    'MultiTaskNN',
    'LogisticRegression',
    
    # Federated Wrappers
    'FederatedModel',
    'create_model_for_task'
]

__version__ = '1.0.0'
__author__ = 'Abbas Yazdinejad and Jude Kong'

# Model registry for easy access
MODEL_REGISTRY = {
    'simple_cnn': SimpleCNN,
    'resnet18': ResNet18PathMNIST,
    'custom_cnn': CustomCNN,
    'tabular_nn': TabularNN,
    'multi_task_nn': MultiTaskNN,
    'logistic_regression': LogisticRegression
}


def get_model(model_name: str, **kwargs):
    """
    Get model by name from registry
    
    Args:
        model_name: Name of the model
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
        
    Example:
        >>> model = get_model('resnet18', num_classes=9)
        >>> model = get_model('tabular_nn', input_dim=10, output_dim=2)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)
