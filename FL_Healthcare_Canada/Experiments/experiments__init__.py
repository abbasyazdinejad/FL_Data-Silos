"""
Demo Experiments module for Federated Learning Healthcare

Contains experiment runners for:
- Cancer Detection (Ontario, Alberta, Quebec)
- PathMNIST Medical Imaging Benchmark
- Master runner for all experiments (Table 5 generation)
"""

# Import experiment runners
try:
    from .exp_cancer_detection import run as run_cancer_detection
    from .exp_pathmnist import run as run_pathmnist
    from .run_all_experiments import main as run_all_experiments
    
    __all__ = [
        'run_cancer_detection',
        'run_pathmnist',
        'run_all_experiments'
    ]
except ImportError:
    # Handle case where experiments are run as standalone scripts
    __all__ = []

__version__ = '1.0.0'
__author__ = 'Abbas Yazdinejad and Jude Kong'
