"""
Example of Federated Learning module for healthcare applications

Implements:
- FedAvg (Federated Averaging)
- FedProx (Federated Proximal) 
- Client-server architecture
- Various aggregation methods
"""

from .aggregation import (
    weighted_average,
    simple_average,
    median_aggregation,
    trimmed_mean_aggregation,
    secure_aggregation,
    aggregate_models,
    compute_model_difference,
    compute_model_norm,
    clip_model_norm
)

from .client import (
    FederatedClient,
    FedProxClient
)

from .server import (
    FederatedServer
)

from .fed_avg import (
    FedAvgTrainer
)

from .fed_prox import (
    FedProxTrainer
)

__all__ = [
    # Aggregation methods
    'weighted_average',
    'simple_average',
    'median_aggregation',
    'trimmed_mean_aggregation',
    'secure_aggregation',
    'aggregate_models',
    'compute_model_difference',
    'compute_model_norm',
    'clip_model_norm',
    
    # Clients
    'FederatedClient',
    'FedProxClient',
    
    # Server
    'FederatedServer',
    
    # Training orchestrators
    'FedAvgTrainer',
    'FedProxTrainer'
]

__version__ = '1.0.0'
__author__ = 'Abbas Yazdinejad and Jude Kong'

# Recommended configurations for Canadian healthcare FL
RECOMMENDED_CONFIG = {
    'cancer_detection': {
        'aggregation_method': 'weighted_avg',
        'num_rounds': 50,
        'local_epochs': 5,
        'learning_rate': 0.001,
        'batch_size': 32,
        'privacy': {'epsilon': 1.0, 'delta': 1e-5}
    },
    'medical_imaging': {
        'aggregation_method': 'weighted_avg',
        'num_rounds': 50,
        'local_epochs': 1,  # Imaging uses fewer local epochs
        'learning_rate': 0.001,
        'batch_size': 32,
        'privacy': {'epsilon': 1.0, 'delta': 1e-5}
    },
    'pandemic_forecasting': {
        'aggregation_method': 'weighted_avg',
        'num_rounds': 100,
        'local_epochs': 5,
        'learning_rate': 0.001,
        'batch_size': 64,
        'privacy': {'epsilon': 1.2, 'delta': 1e-5}
    }
}


def get_recommended_config(use_case: str) -> dict:
    """
    Get recommended FL configuration for a use case
    
    Args:
        use_case: One of 'cancer_detection', 'medical_imaging', 'pandemic_forecasting'
        
    Returns:
        Configuration dictionary
    """
    if use_case not in RECOMMENDED_CONFIG:
        raise ValueError(f"Unknown use case: {use_case}. "
                        f"Available: {list(RECOMMENDED_CONFIG.keys())}")
    
    return RECOMMENDED_CONFIG[use_case].copy()
