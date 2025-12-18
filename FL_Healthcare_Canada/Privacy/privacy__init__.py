"""
Demo of Privacy module for Differential Privacy in Federated Learning

Implements (ε, δ)-differential privacy:
- Gradient clipping
- Noise mechanisms (Gaussian, Laplace)
- DP-SGD optimizer
- Privacy accounting (RDP)

Based on:
- Abadi et al. 2016: Deep Learning with Differential Privacy
- Mironov 2017: Rényi Differential Privacy
"""

from .differential_privacy import (
    DifferentialPrivacy,
    apply_dp_to_model_update,
    compute_privacy_loss
)

from .privacy_engine import (
    PrivacyEngine,
    PrivateOptimizer
)

__all__ = [
    # Core DP class
    'DifferentialPrivacy',
    
    # Privacy engine
    'PrivacyEngine',
    'PrivateOptimizer',
    
    # Utility functions
    'apply_dp_to_model_update',
    'compute_privacy_loss'
]

__version__ = '1.0.0'
__author__ = 'Abbas Yazdinejad and Jude Kong'

# Paper privacy parameters (Table 5)
PAPER_PRIVACY_PARAMS = {
    'cancer_detection': {
        'epsilon': 1.0,
        'delta': 1e-5,
        'noise_multiplier': 1.0,
        'max_grad_norm': 1.0,
        'mechanism': 'gaussian'
    },
    'medical_imaging': {
        'epsilon': 1.0,
        'delta': 1e-5,
        'noise_multiplier': 1.0,
        'max_grad_norm': 1.0,
        'mechanism': 'gaussian'
    },
    'pandemic_forecasting': {
        'epsilon': 1.2,
        'delta': 1e-5,
        'noise_multiplier': 0.9,
        'max_grad_norm': 1.0,
        'mechanism': 'gaussian'
    }
}


def get_recommended_privacy_config(use_case: str) -> dict:
    """
    Get recommended privacy configuration for a use case
    
    Args:
        use_case: One of 'cancer_detection', 'medical_imaging', 'pandemic_forecasting'
        
    Returns:
        Privacy configuration dict
    """
    if use_case not in PAPER_PRIVACY_PARAMS:
        raise ValueError(f"Unknown use case: {use_case}. "
                        f"Available: {list(PAPER_PRIVACY_PARAMS.keys())}")
    
    return PAPER_PRIVACY_PARAMS[use_case].copy()


def create_privacy_engine(
    model,
    optimizer,
    use_case: str = 'cancer_detection',
    custom_config: dict = None
) -> PrivacyEngine:
    """
    Create privacy engine with recommended or custom configuration
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        use_case: Use case name (for recommended config)
        custom_config: Optional custom configuration (overrides use_case)
        
    Returns:
        Configured PrivacyEngine
    """
    if custom_config:
        config = custom_config
    else:
        config = get_recommended_privacy_config(use_case)
    
    return PrivacyEngine(
        model=model,
        optimizer=optimizer,
        noise_multiplier=config['noise_multiplier'],
        max_grad_norm=config['max_grad_norm'],
        delta=config['delta'],
        target_epsilon=config['epsilon']
    )


def validate_privacy_parameters(
    epsilon: float,
    delta: float,
    dataset_size: int
) -> Dict[str, any]:
    """
    Validate privacy parameters meet standards
    
    Args:
        epsilon: Privacy parameter
        delta: Privacy parameter  
        dataset_size: Dataset size
        
    Returns:
        Validation results
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check epsilon
    if epsilon > 10.0:
        validation['errors'].append(f"Epsilon too high: {epsilon} > 10.0 (weak privacy)")
        validation['valid'] = False
    elif epsilon > 2.0:
        validation['warnings'].append(f"Epsilon high: {epsilon} > 2.0 (consider reducing)")
    
    # Check delta
    if delta > 1 / dataset_size:
        validation['errors'].append(f"Delta too high: {delta} > 1/{dataset_size}")
        validation['valid'] = False
    
    # Check if delta is appropriate
    if delta > 1e-3:
        validation['warnings'].append(f"Delta large: {delta} > 1e-3")
    
    return validation


# Privacy levels for interpretation
PRIVACY_LEVELS = {
    'very_strong': (0.0, 0.1),
    'strong': (0.1, 1.0),
    'moderate': (1.0, 10.0),
    'weak': (10.0, float('inf'))
}


def interpret_privacy_level(epsilon: float) -> str:
    """
    Interpret privacy level from epsilon
    
    Args:
        epsilon: Privacy parameter
        
    Returns:
        Privacy level string
    """
    for level, (low, high) in PRIVACY_LEVELS.items():
        if low <= epsilon < high:
            return level
    return 'unknown'


def print_privacy_summary(
    epsilon: float,
    delta: float,
    noise_multiplier: float,
    max_grad_norm: float
):
    """
    Print human-readable privacy summary
    
    Args:
        epsilon: Privacy parameter
        delta: Privacy parameter
        noise_multiplier: Noise scale
        max_grad_norm: Clipping threshold
    """
    level = interpret_privacy_level(epsilon)
    
    print("="*70)
    print("DIFFERENTIAL PRIVACY SUMMARY")
    print("="*70)
    print(f"Privacy Parameters:")
    print(f"  Epsilon (ε): {epsilon:.4f}")
    print(f"  Delta (δ): {delta:.2e}")
    print(f"  Privacy Level: {level.replace('_', ' ').title()}")
    print(f"\nMechanism Parameters:")
    print(f"  Noise Multiplier (σ): {noise_multiplier:.4f}")
    print(f"  Max Gradient Norm (C): {max_grad_norm:.4f}")
    print(f"  Mechanism: Gaussian")
    print(f"\nInterpretation:")
    if epsilon < 1.0:
        print(f"  ✓ Strong privacy guarantee (ε < 1.0)")
    elif epsilon < 2.0:
        print(f"  ✓ Good privacy guarantee (ε < 2.0)")
    elif epsilon < 10.0:
        print(f"  ⚠ Moderate privacy guarantee (ε < 10.0)")
    else:
        print(f"  ✗ Weak privacy guarantee (ε ≥ 10.0)")
    print("="*70)


# Example usage
def example_usage():
    """Example of using the privacy module"""
    import torch
    import torch.nn as nn
    
    # Create model and optimizer
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create privacy engine with recommended config
    privacy_engine = create_privacy_engine(
        model, optimizer, use_case='cancer_detection'
    )
    
    # Print configuration
    print_privacy_summary(
        epsilon=1.0,
        delta=1e-5,
        noise_multiplier=1.0,
        max_grad_norm=1.0
    )


if __name__ == "__main__":
    example_usage()
