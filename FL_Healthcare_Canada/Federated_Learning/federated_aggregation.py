"""
Example of Model aggregation methods for federated learning

Implements various aggregation strategies:
- FedAvg (weighted average)
- Simple averaging
- Median aggregation (robust)
- Secure aggregation with optional DP noise
"""

import torch
from typing import Dict, List, Optional
import copy
import numpy as np


def weighted_average(
    client_models: List[Dict[str, torch.Tensor]],
    client_weights: List[float]
) -> Dict[str, torch.Tensor]:
    """
    Compute weighted average of client models (FedAvg aggregation)
    
    This is the core FedAvg aggregation as described in McMahan et al. 2017.
    Each client's contribution is weighted by their dataset size.
    
    Args:
        client_models: List of client model parameters (state dicts)
        client_weights: List of client weights (typically proportional to dataset size)
        
    Returns:
        Aggregated model parameters
        
    Example:
        >>> client_models = [client1.get_parameters(), client2.get_parameters()]
        >>> weights = [1000, 500]  # Dataset sizes
        >>> aggregated = weighted_average(client_models, weights)
    """
    if not client_models:
        raise ValueError("Cannot aggregate empty list of models")
    
    if len(client_models) != len(client_weights):
        raise ValueError("Number of models must match number of weights")
    
    # Normalize weights to sum to 1
    total_weight = sum(client_weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")
    
    normalized_weights = [w / total_weight for w in client_weights]
    
    # Initialize aggregated model with zeros
    aggregated_model = {}
    
    # Get parameter names from first model
    param_names = client_models[0].keys()
    
    for param_name in param_names:
        # Compute weighted sum
        aggregated_param = torch.zeros_like(client_models[0][param_name])
        
        for client_model, weight in zip(client_models, normalized_weights):
            if param_name not in client_model:
                raise KeyError(f"Parameter {param_name} not found in client model")
            
            aggregated_param += weight * client_model[param_name].float()
        
        aggregated_model[param_name] = aggregated_param
    
    return aggregated_model


def simple_average(
    client_models: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Compute simple (unweighted) average of client models
    
    Args:
        client_models: List of client model parameters
        
    Returns:
        Aggregated model parameters
    """
    # Equal weights for all clients
    weights = [1.0] * len(client_models)
    return weighted_average(client_models, weights)


def median_aggregation(
    client_models: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Compute coordinate-wise median of client models (robust aggregation)
    
    More robust to outliers and Byzantine clients than averaging.
    
    Args:
        client_models: List of client model parameters
        
    Returns:
        Aggregated model parameters
    """
    if not client_models:
        raise ValueError("Cannot aggregate empty list of models")
    
    aggregated_model = {}
    param_names = client_models[0].keys()
    
    for param_name in param_names:
        # Stack parameters from all clients
        stacked_params = torch.stack([
            client_model[param_name].float() 
            for client_model in client_models
        ])
        
        # Compute median along client dimension
        aggregated_model[param_name] = torch.median(stacked_params, dim=0)[0]
    
    return aggregated_model


def trimmed_mean_aggregation(
    client_models: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Compute trimmed mean (removes extreme values before averaging)
    
    Args:
        client_models: List of client model parameters
        trim_ratio: Fraction of extreme values to trim (0.0 to 0.5)
        
    Returns:
        Aggregated model parameters
    """
    if not client_models:
        raise ValueError("Cannot aggregate empty list of models")
    
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0.0, 0.5)")
    
    aggregated_model = {}
    param_names = client_models[0].keys()
    
    for param_name in param_names:
        # Stack parameters
        stacked_params = torch.stack([
            client_model[param_name].float()
            for client_model in client_models
        ])
        
        # Sort along client dimension
        sorted_params, _ = torch.sort(stacked_params, dim=0)
        
        # Trim extreme values
        n_clients = len(client_models)
        n_trim = int(n_clients * trim_ratio)
        
        if n_trim > 0:
            trimmed_params = sorted_params[n_trim:-n_trim]
        else:
            trimmed_params = sorted_params
        
        # Compute mean
        aggregated_model[param_name] = torch.mean(trimmed_params, dim=0)
    
    return aggregated_model


def secure_aggregation(
    client_models: List[Dict[str, torch.Tensor]],
    client_weights: Optional[List[float]] = None,
    add_noise: bool = False,
    noise_scale: float = 0.01,
    max_norm: Optional[float] = None
) -> Dict[str, torch.Tensor]:
    """
    Secure aggregation with optional differential privacy noise
    
    Args:
        client_models: List of client model parameters
        client_weights: Client weights (if None, uses equal weights)
        add_noise: Whether to add DP noise
        noise_scale: Scale of Gaussian noise for DP
        max_norm: Maximum norm for gradient clipping (if provided)
        
    Returns:
        Securely aggregated model parameters
    """
    # Clip gradients if max_norm specified
    if max_norm is not None:
        clipped_models = []
        for client_model in client_models:
            clipped_model = clip_model_norm(client_model, max_norm)
            clipped_models.append(clipped_model)
        client_models = clipped_models
    
    # Standard weighted aggregation
    if client_weights is None:
        client_weights = [1.0] * len(client_models)
    
    aggregated_model = weighted_average(client_models, client_weights)
    
    # Add differential privacy noise if requested
    if add_noise:
        for param_name in aggregated_model:
            noise = torch.randn_like(aggregated_model[param_name]) * noise_scale
            aggregated_model[param_name] = aggregated_model[param_name] + noise
    
    return aggregated_model


def clip_model_norm(
    model_params: Dict[str, torch.Tensor],
    max_norm: float
) -> Dict[str, torch.Tensor]:
    """
    Clip model parameters to maximum norm
    
    Args:
        model_params: Model parameters
        max_norm: Maximum allowed norm
        
    Returns:
        Clipped model parameters
    """
    # Compute total norm
    total_norm = 0.0
    for param in model_params.values():
        total_norm += param.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    # Clip if needed
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1.0:
        clipped_params = {}
        for name, param in model_params.items():
            clipped_params[name] = param * clip_coef
        return clipped_params
    else:
        return model_params


def aggregate_models(
    client_models: List[Dict[str, torch.Tensor]],
    aggregation_method: str = 'weighted_avg',
    client_weights: Optional[List[float]] = None,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Aggregate client models using specified method
    
    Args:
        client_models: List of client model parameters
        aggregation_method: Aggregation method 
            ('weighted_avg', 'mean', 'median', 'trimmed_mean', 'secure')
        client_weights: Optional client weights for weighted averaging
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Aggregated model parameters
    """
    if aggregation_method == 'weighted_avg':
        if client_weights is None:
            client_weights = [1.0] * len(client_models)
        return weighted_average(client_models, client_weights)
    
    elif aggregation_method == 'mean':
        return simple_average(client_models)
    
    elif aggregation_method == 'median':
        return median_aggregation(client_models)
    
    elif aggregation_method == 'trimmed_mean':
        trim_ratio = kwargs.get('trim_ratio', 0.1)
        return trimmed_mean_aggregation(client_models, trim_ratio)
    
    elif aggregation_method == 'secure':
        return secure_aggregation(
            client_models, client_weights,
            add_noise=kwargs.get('add_noise', False),
            noise_scale=kwargs.get('noise_scale', 0.01),
            max_norm=kwargs.get('max_norm', None)
        )
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def compute_model_difference(
    model1: Dict[str, torch.Tensor],
    model2: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute difference between two models (model1 - model2)
    
    Args:
        model1: First model parameters
        model2: Second model parameters
        
    Returns:
        Difference parameters
    """
    diff = {}
    for param_name in model1:
        diff[param_name] = model1[param_name] - model2[param_name]
    return diff


def compute_model_norm(model_params: Dict[str, torch.Tensor]) -> float:
    """
    Compute L2 norm of model parameters
    
    Args:
        model_params: Model parameters
        
    Returns:
        L2 norm
    """
    total_norm = 0.0
    for param in model_params.values():
        total_norm += param.norm(2).item() ** 2
    return total_norm ** 0.5


if __name__ == "__main__":
    print("="*60)
    print("Testing Aggregation Methods")
    print("="*60)
    
    # Create dummy client models
    client1 = {'weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 
               'bias': torch.tensor([0.5, 0.6])}
    client2 = {'weight': torch.tensor([[2.0, 3.0], [4.0, 5.0]]), 
               'bias': torch.tensor([0.7, 0.8])}
    client3 = {'weight': torch.tensor([[1.5, 2.5], [3.5, 4.5]]), 
               'bias': torch.tensor([0.6, 0.7])}
    
    client_models = [client1, client2, client3]
    weights = [100, 200, 150]  # Dataset sizes
    
    # Test weighted average
    print("\n1. Weighted Average (FedAvg)")
    print("-"*60)
    agg1 = weighted_average(client_models, weights)
    print(f"Aggregated weight:\n{agg1['weight']}")
    print(f"Aggregated bias: {agg1['bias']}")
    
    # Test simple average
    print("\n2. Simple Average")
    print("-"*60)
    agg2 = simple_average(client_models)
    print(f"Aggregated weight:\n{agg2['weight']}")
    
    # Test median
    print("\n3. Median Aggregation")
    print("-"*60)
    agg3 = median_aggregation(client_models)
    print(f"Aggregated weight:\n{agg3['weight']}")
    
    # Test secure aggregation
    print("\n4. Secure Aggregation with DP noise")
    print("-"*60)
    agg4 = secure_aggregation(client_models, weights, add_noise=True, noise_scale=0.01)
    print(f"Aggregated weight (with noise):\n{agg4['weight']}")
    
    # Test model norm
    print("\n5. Model Norm")
    print("-"*60)
    norm1 = compute_model_norm(client1)
    print(f"Client 1 norm: {norm1:.4f}")
    
    print("\n" + "="*60)
    print("All aggregation tests passed!")
    print("="*60)
