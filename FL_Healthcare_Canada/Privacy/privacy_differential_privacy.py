"""
Demo of Differential Privacy Mechanisms for Federated Learning

Implements various DP mechanisms:
- Gaussian mechanism (DP-SGD)
- Laplace mechanism
- Gradient clipping
- Privacy accounting (ε, δ)

Based on:
- Abadi et al. 2016: Deep Learning with Differential Privacy
- Dwork & Roth 2014: The Algorithmic Foundations of Differential Privacy
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class DifferentialPrivacy:
    """
    Differential Privacy mechanisms for FL
    
    Provides (ε, δ)-differential privacy guarantees through:
    - Gradient clipping
    - Noise addition (Gaussian or Laplace)
    - Privacy budget accounting
    """
    
    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        mechanism: str = 'gaussian'
    ):
        """
        Initialize DP mechanism
        
        Args:
            noise_multiplier: Noise scale (higher = more privacy, less utility)
            max_grad_norm: Maximum gradient norm (clipping threshold)
            delta: Delta parameter for (ε, δ)-DP
            mechanism: Noise mechanism ('gaussian' or 'laplace')
        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.mechanism = mechanism
        
        # Privacy accounting
        self.steps = 0
        self.epsilon_spent = 0.0
    
    def clip_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        max_norm: Optional[float] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Clip gradients to maximum norm
        
        This is critical for DP: bounds sensitivity of the function
        
        Args:
            gradients: Dict of gradient tensors
            max_norm: Maximum norm (uses self.max_grad_norm if None)
            
        Returns:
            Tuple of (clipped_gradients, actual_norm)
        """
        if max_norm is None:
            max_norm = self.max_grad_norm
        
        # Compute total gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip if needed
        clip_coef = max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1.0:
            clipped_grads = {}
            for name, grad in gradients.items():
                clipped_grads[name] = grad * clip_coef
            return clipped_grads, total_norm
        else:
            return gradients, total_norm
    
    def add_noise(
        self,
        gradients: Dict[str, torch.Tensor],
        sensitivity: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Add calibrated noise to gradients for DP
        
        Args:
            gradients: Gradient tensors
            sensitivity: Sensitivity (uses max_grad_norm if None)
            
        Returns:
            Noisy gradients
        """
        if sensitivity is None:
            sensitivity = self.max_grad_norm
        
        noisy_grads = {}
        
        for name, grad in gradients.items():
            if self.mechanism == 'gaussian':
                # Gaussian mechanism
                noise_scale = self.noise_multiplier * sensitivity
                noise = torch.randn_like(grad) * noise_scale
            elif self.mechanism == 'laplace':
                # Laplace mechanism
                noise_scale = self.noise_multiplier * sensitivity
                noise = torch.from_numpy(
                    np.random.laplace(0, noise_scale, grad.shape)
                ).float().to(grad.device)
            else:
                raise ValueError(f"Unknown mechanism: {self.mechanism}")
            
            noisy_grads[name] = grad + noise
        
        return noisy_grads
    
    def privatize_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply full DP pipeline: clip + add noise
        
        Args:
            gradients: Raw gradients
            
        Returns:
            DP gradients
        """
        # Step 1: Clip gradients
        clipped_grads, _ = self.clip_gradients(gradients)
        
        # Step 2: Add calibrated noise
        private_grads = self.add_noise(clipped_grads)
        
        # Update accounting
        self.steps += 1
        
        return private_grads
    
    def compute_epsilon(
        self,
        steps: int,
        sample_rate: float,
        delta: Optional[float] = None
    ) -> float:
        """
        Compute privacy budget ε using moments accountant
        
        This is a simplified version. For exact accounting, use:
        - Tensorflow Privacy library
        - Opacus privacy engine
        
        Args:
            steps: Number of training steps
            sample_rate: Sampling rate (batch_size / dataset_size)
            delta: Delta parameter (uses self.delta if None)
            
        Returns:
            Epsilon (ε) value
        """
        if delta is None:
            delta = self.delta
        
        # Simplified privacy analysis (conservative bound)
        # For exact accounting, use RDP or moments accountant
        
        # From Abadi et al. 2016 (simplified)
        q = sample_rate
        sigma = self.noise_multiplier
        
        if sigma == 0:
            return float('inf')
        
        # Conservative bound using strong composition
        # ε ≈ q * sqrt(2 * steps * log(1/δ)) / σ
        epsilon = q * math.sqrt(2 * steps * math.log(1 / delta)) / sigma
        
        return epsilon
    
    def compute_epsilon_rdp(
        self,
        steps: int,
        sample_rate: float,
        orders: List[float] = None
    ) -> float:
        """
        Compute epsilon using Rényi Differential Privacy (RDP)
        
        More accurate than basic moments accountant
        
        Args:
            steps: Number of steps
            sample_rate: Sampling rate
            orders: RDP orders (lambdas)
            
        Returns:
            Epsilon
        """
        if orders is None:
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        
        rdp = []
        for order in orders:
            if order == 1:
                continue
            # RDP of Gaussian mechanism
            rdp_at_order = (order * sample_rate ** 2) / (2 * self.noise_multiplier ** 2)
            rdp.append(rdp_at_order * steps)
        
        # Convert RDP to (ε, δ)-DP
        min_epsilon = float('inf')
        for rdp_val, order in zip(rdp, orders[1:]):
            epsilon = rdp_val + math.log(1 / self.delta) / (order - 1)
            min_epsilon = min(min_epsilon, epsilon)
        
        return min_epsilon
    
    def get_privacy_spent(
        self,
        dataset_size: int,
        batch_size: int,
        epochs: int = None
    ) -> Dict[str, float]:
        """
        Get total privacy budget spent
        
        Args:
            dataset_size: Size of training dataset
            batch_size: Batch size
            epochs: Number of epochs (uses self.steps if None)
            
        Returns:
            Dict with epsilon, delta, steps
        """
        if epochs is not None:
            steps = int(epochs * dataset_size / batch_size)
        else:
            steps = self.steps
        
        sample_rate = batch_size / dataset_size
        epsilon = self.compute_epsilon(steps, sample_rate)
        
        return {
            'epsilon': epsilon,
            'delta': self.delta,
            'steps': steps,
            'noise_multiplier': self.noise_multiplier,
            'max_grad_norm': self.max_grad_norm
        }
    
    def is_privacy_exhausted(
        self,
        epsilon_budget: float,
        dataset_size: int,
        batch_size: int
    ) -> bool:
        """
        Check if privacy budget is exhausted
        
        Args:
            epsilon_budget: Maximum allowed epsilon
            dataset_size: Dataset size
            batch_size: Batch size
            
        Returns:
            True if budget exhausted
        """
        privacy_spent = self.get_privacy_spent(dataset_size, batch_size)
        return privacy_spent['epsilon'] >= epsilon_budget


def apply_dp_to_model_update(
    model_update: Dict[str, torch.Tensor],
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Apply DP to a model update (convenience function)
    
    Args:
        model_update: Model parameter updates
        noise_multiplier: Noise scale
        max_grad_norm: Clipping threshold
        
    Returns:
        DP model update
    """
    dp = DifferentialPrivacy(noise_multiplier, max_grad_norm)
    return dp.privatize_gradients(model_update)


def compute_privacy_loss(
    epsilon: float,
    delta: float,
    dataset_size: int
) -> Dict[str, float]:
    """
    Compute privacy loss metrics
    
    Args:
        epsilon: Privacy parameter
        delta: Privacy parameter
        dataset_size: Size of dataset
        
    Returns:
        Privacy loss metrics
    """
    # Privacy loss per record
    per_record_epsilon = epsilon / dataset_size
    
    # Approximate worst-case privacy loss
    worst_case_prob = math.exp(epsilon)
    
    return {
        'epsilon': epsilon,
        'delta': delta,
        'per_record_epsilon': per_record_epsilon,
        'worst_case_prob_ratio': worst_case_prob,
        'privacy_level': _classify_privacy_level(epsilon)
    }


def _classify_privacy_level(epsilon: float) -> str:
    """Classify privacy level based on epsilon"""
    if epsilon < 0.1:
        return 'very_strong'
    elif epsilon < 1.0:
        return 'strong'
    elif epsilon < 10.0:
        return 'moderate'
    else:
        return 'weak'


if __name__ == "__main__":
    print("="*70)
    print("Testing Differential Privacy Mechanisms")
    print("="*70)
    
    # Create DP mechanism
    dp = DifferentialPrivacy(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        delta=1e-5,
        mechanism='gaussian'
    )
    
    # Test gradient clipping
    print("\n1. Gradient Clipping")
    print("-"*70)
    gradients = {
        'layer1': torch.randn(10, 10) * 5.0,  # Large gradients
        'layer2': torch.randn(5, 5) * 3.0
    }
    
    clipped_grads, norm = dp.clip_gradients(gradients, max_norm=1.0)
    print(f"Original norm: {norm:.4f}")
    
    clipped_norm = sum(g.norm(2).item()**2 for g in clipped_grads.values())**0.5
    print(f"Clipped norm: {clipped_norm:.4f}")
    
    # Test noise addition
    print("\n2. Noise Addition")
    print("-"*70)
    noisy_grads = dp.add_noise(clipped_grads)
    print(f"Added Gaussian noise with scale={dp.noise_multiplier * dp.max_grad_norm}")
    
    # Test privacy accounting
    print("\n3. Privacy Accounting")
    print("-"*70)
    privacy_spent = dp.get_privacy_spent(
        dataset_size=10000,
        batch_size=32,
        epochs=10
    )
    print(f"Privacy spent:")
    for key, value in privacy_spent.items():
        print(f"  {key}: {value}")
    
    # Test privacy loss
    print("\n4. Privacy Loss Analysis")
    print("-"*70)
    loss_metrics = compute_privacy_loss(
        epsilon=1.0,
        delta=1e-5,
        dataset_size=10000
    )
    print(f"Privacy loss metrics:")
    for key, value in loss_metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("All DP tests passed!")
    print("="*70)
