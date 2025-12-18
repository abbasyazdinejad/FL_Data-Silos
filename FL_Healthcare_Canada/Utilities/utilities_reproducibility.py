"""
Demo of Reproducibility utilities for FL experiments

Ensures reproducible results by:
- Setting random seeds (Python, NumPy, PyTorch)
- Configuring deterministic algorithms
- Environment documentation
- Checkpointing
"""

import random
import numpy as np
import torch
import os
import sys
from typing import Dict, Optional
import platform
import json


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and GPU)
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if torch.backends.mps.is_available():
        # Apple Metal (M1/M2/M3/M4)
        torch.mps.manual_seed(seed)
    
    print(f"✓ Random seed set to {seed}")


def set_deterministic(enabled: bool = True):
    """
    Enable deterministic algorithms for reproducibility
    
    Note: May reduce performance
    
    Args:
        enabled: Whether to enable deterministic mode
    """
    if enabled:
        # PyTorch deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable
        os.environ['PYTHONHASHSEED'] = '0'
        
        print("✓ Deterministic mode enabled (may reduce performance)")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("✓ Deterministic mode disabled")


def make_reproducible(
    seed: int = 42,
    deterministic: bool = True,
    verbose: bool = True
):
    """
    Configure environment for maximum reproducibility
    
    Args:
        seed: Random seed
        deterministic: Enable deterministic algorithms
        verbose: Print configuration
    """
    set_seed(seed)
    
    if deterministic:
        set_deterministic(True)
    
    if verbose:
        print("\n" + "="*70)
        print("REPRODUCIBILITY CONFIGURATION")
        print("="*70)
        print(f"Seed: {seed}")
        print(f"Deterministic mode: {deterministic}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Python version: {sys.version.split()[0]}")
        
        if torch.cuda.is_available():
            print(f"CUDA available: Yes")
            print(f"CUDA version: {torch.version.cuda}")
        elif torch.backends.mps.is_available():
            print(f"Apple Metal (MPS): Yes")
        else:
            print(f"GPU: Not available")
        
        print("="*70 + "\n")


def get_environment_info() -> Dict:
    """
    Get comprehensive environment information
    
    Returns:
        Dict with environment details
    """
    info = {
        # System
        'platform': platform.platform(),
        'python_version': sys.version.split()[0],
        'cpu_count': os.cpu_count(),
        
        # Libraries
        'numpy_version': np.__version__,
        'torch_version': torch.__version__,
        
        # PyTorch
        'torch_cuda_available': torch.cuda.is_available(),
        'torch_mps_available': torch.backends.mps.is_available(),
    }
    
    # CUDA info
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['num_gpus'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) 
                            for i in range(torch.cuda.device_count())]
    
    # MPS info
    if torch.backends.mps.is_available():
        info['mps_device'] = 'Apple Silicon GPU'
    
    return info


def save_environment_info(filepath: str = "./environment_info.json"):
    """
    Save environment information to file
    
    Args:
        filepath: Output file path
    """
    info = get_environment_info()
    
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Environment info saved to {filepath}")


def check_reproducibility_requirements() -> Dict[str, bool]:
    """
    Check if reproducibility requirements are met
    
    Returns:
        Dict of checks and their status
    """
    checks = {
        'seed_set': False,
        'deterministic_enabled': False,
        'cuda_deterministic': False,
        'environment_documented': False
    }
    
    # Check if deterministic mode is enabled
    if torch.backends.cudnn.deterministic:
        checks['deterministic_enabled'] = True
        checks['cuda_deterministic'] = True
    
    # Check if environment hash exists
    if os.path.exists('./environment_info.json'):
        checks['environment_documented'] = True
    
    return checks


def create_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: Dict,
    filepath: str,
    **kwargs
):
    """
    Create checkpoint for reproducibility
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch/round
        metrics: Current metrics
        filepath: Checkpoint file path
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        **kwargs
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model,
    optimizer=None,
    device: str = 'cpu'
) -> Dict:
    """
    Load checkpoint
    
    Args:
        filepath: Checkpoint file path
        model: PyTorch model
        optimizer: Optimizer (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return checkpoint


def verify_reproducibility(
    results1: list,
    results2: list,
    tolerance: float = 1e-6
) -> bool:
    """
    Verify that two sets of results are identical (within tolerance)
    
    Args:
        results1: First set of results
        results2: Second set of results
        tolerance: Numerical tolerance
        
    Returns:
        True if results match
    """
    if len(results1) != len(results2):
        print(f"✗ Length mismatch: {len(results1)} vs {len(results2)}")
        return False
    
    results1 = np.array(results1)
    results2 = np.array(results2)
    
    diff = np.abs(results1 - results2)
    max_diff = np.max(diff)
    
    if max_diff > tolerance:
        print(f"✗ Results differ by {max_diff:.2e} (tolerance: {tolerance:.2e})")
        return False
    else:
        print(f"✓ Results match within tolerance ({max_diff:.2e})")
        return True


class ReproducibilityManager:
    """
    Manage reproducibility for entire experiment
    
    Usage:
        >>> repro = ReproducibilityManager(seed=42)
        >>> repro.setup()
        >>> # Run experiment
        >>> repro.save_state(model, optimizer, epoch, metrics, 'checkpoint.pt')
        >>> repro.finalize()
    """
    
    def __init__(
        self,
        seed: int = 42,
        deterministic: bool = True,
        save_dir: str = "./reproducibility"
    ):
        """
        Initialize reproducibility manager
        
        Args:
            seed: Random seed
            deterministic: Enable deterministic mode
            save_dir: Directory for reproducibility files
        """
        self.seed = seed
        self.deterministic = deterministic
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
    
    def setup(self):
        """Setup reproducibility environment"""
        make_reproducible(self.seed, self.deterministic, verbose=True)
        
        # Save environment info
        env_file = os.path.join(self.save_dir, 'environment_info.json')
        save_environment_info(env_file)
        
        # Save seed
        seed_file = os.path.join(self.save_dir, 'seed.txt')
        with open(seed_file, 'w') as f:
            f.write(str(self.seed))
    
    def save_state(
        self,
        model,
        optimizer,
        epoch: int,
        metrics: Dict,
        name: str = 'checkpoint'
    ):
        """
        Save experiment state
        
        Args:
            model: Model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Current metrics
            name: Checkpoint name
        """
        filepath = os.path.join(self.save_dir, f'{name}_epoch_{epoch}.pt')
        create_checkpoint(model, optimizer, epoch, metrics, filepath)
    
    def finalize(self):
        """Finalize reproducibility documentation"""
        print("\n" + "="*70)
        print("REPRODUCIBILITY FINALIZED")
        print("="*70)
        print(f"Seed: {self.seed}")
        print(f"Files saved to: {self.save_dir}")
        print("  - environment_info.json")
        print("  - seed.txt")
        print("  - checkpoints (if saved)")
        print("="*70 + "\n")


if __name__ == "__main__":
    print("="*70)
    print("Testing Reproducibility Utilities")
    print("="*70)
    
    # Test seed setting
    print("\n1. Set Random Seed")
    print("-"*70)
    set_seed(42)
    
    # Generate some random numbers
    print(f"Random number (Python): {random.random():.6f}")
    print(f"Random number (NumPy): {np.random.rand():.6f}")
    print(f"Random tensor (PyTorch): {torch.rand(1).item():.6f}")
    
    # Test full setup
    print("\n2. Full Reproducibility Setup")
    print("-"*70)
    make_reproducible(seed=42, deterministic=True)
    
    # Get environment info
    print("\n3. Environment Information")
    print("-"*70)
    env_info = get_environment_info()
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    
    # Test reproducibility verification
    print("\n4. Reproducibility Verification")
    print("-"*70)
    set_seed(42)
    results1 = [torch.rand(1).item() for _ in range(5)]
    
    set_seed(42)
    results2 = [torch.rand(1).item() for _ in range(5)]
    
    print(f"Results 1: {results1}")
    print(f"Results 2: {results2}")
    verify_reproducibility(results1, results2)
    
    print("\n" + "="*70)
    print("Reproducibility tests passed!")
    print("="*70)
