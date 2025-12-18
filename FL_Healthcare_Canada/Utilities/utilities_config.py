"""
Demo of Configuration management for FL Healthcare experiments
Provides centralized configuration for:
- Experiment parameters
- Model hyperparameters
- Privacy settings
- Governance settings
- Paths and directories
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment"""
    # Experiment info
    name: str = "cancer_detection"
    use_case: str = "cancer_detection"
    seed: int = 42
    
    # Data settings
    data_path: str = "./data"
    batch_size: int = 32
    test_split: float = 0.2
    
    # Federated learning
    num_rounds: int = 50
    local_epochs: int = 5
    num_clients: int = 3
    aggregation_method: str = "weighted_avg"
    
    # Model settings
    model_name: str = "TabularNN"
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Privacy settings
    enable_privacy: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    
    # Governance settings
    enable_consent: bool = True
    enable_audit: bool = True
    enable_fairness: bool = True
    enable_compliance: bool = True
    
    # Evaluation
    evaluate_every: int = 1
    num_repetitions: int = 5
    
    # Output
    output_dir: str = "./results"
    save_checkpoints: bool = True
    verbose: bool = True
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"


class ConfigManager:
    """
    Centralized configuration management
    
    Handles loading, saving, and validating experiment configurations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager
        
        Args:
            config_path: Path to config file (optional)
        """
        self.config_path = config_path
        self.config = None
        
        if config_path and os.path.exists(config_path):
            self.load(config_path)
        else:
            self.config = ExperimentConfig()
    
    def load(self, path: str):
        """
        Load configuration from file
        
        Args:
            path: Path to config file (.json or .yaml)
        """
        ext = Path(path).suffix.lower()
        
        with open(path, 'r') as f:
            if ext == '.json':
                config_dict = json.load(f)
            elif ext in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {ext}")
        
        self.config = ExperimentConfig(**config_dict)
        print(f"✓ Loaded config from {path}")
    
    def save(self, path: str):
        """
        Save configuration to file
        
        Args:
            path: Output path (.json or .yaml)
        """
        if self.config is None:
            raise ValueError("No configuration to save")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        config_dict = asdict(self.config)
        ext = Path(path).suffix.lower()
        
        with open(path, 'w') as f:
            if ext == '.json':
                json.dump(config_dict, f, indent=2)
            elif ext in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {ext}")
        
        print(f"✓ Saved config to {path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if self.config is None:
            return default
        return getattr(self.config, key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        if self.config is None:
            self.config = ExperimentConfig()
        setattr(self.config, key, value)
    
    def update(self, **kwargs):
        """Update multiple configuration values"""
        for key, value in kwargs.items():
            self.set(key, value)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        if self.config is None:
            return {}
        return asdict(self.config)
    
    def print_config(self):
        """Print configuration in readable format"""
        if self.config is None:
            print("No configuration loaded")
            return
        
        print("\n" + "="*70)
        print("EXPERIMENT CONFIGURATION")
        print("="*70)
        
        config_dict = asdict(self.config)
        
        sections = {
            'Experiment': ['name', 'use_case', 'seed'],
            'Data': ['data_path', 'batch_size', 'test_split'],
            'Federated Learning': ['num_rounds', 'local_epochs', 'num_clients', 'aggregation_method'],
            'Model': ['model_name', 'learning_rate', 'optimizer'],
            'Privacy': ['enable_privacy', 'epsilon', 'delta', 'noise_multiplier', 'max_grad_norm'],
            'Governance': ['enable_consent', 'enable_audit', 'enable_fairness', 'enable_compliance'],
            'Evaluation': ['evaluate_every', 'num_repetitions'],
            'Output': ['output_dir', 'save_checkpoints', 'verbose'],
            'Device': ['device']
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                if key in config_dict:
                    print(f"  {key}: {config_dict[key]}")
        
        print("="*70 + "\n")


# Predefined configurations for each use case
PREDEFINED_CONFIGS = {
    'cancer_detection': {
        'name': 'cancer_detection',
        'use_case': 'cancer_detection',
        'model_name': 'TabularNN',
        'num_rounds': 50,
        'local_epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epsilon': 1.0,
        'delta': 1e-5
    },
    'medical_imaging': {
        'name': 'pathmnist',
        'use_case': 'medical_imaging',
        'model_name': 'ResNet18PathMNIST',
        'num_rounds': 50,
        'local_epochs': 1,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epsilon': 1.0,
        'delta': 1e-5
    },
    'pandemic_forecasting': {
        'name': 'pandemic',
        'use_case': 'pandemic_forecasting',
        'model_name': 'LSTM',
        'num_rounds': 100,
        'local_epochs': 5,
        'batch_size': 64,
        'learning_rate': 0.001,
        'epsilon': 1.2,
        'delta': 1e-5
    }
}


def get_config(use_case: str) -> ExperimentConfig:
    """
    Get predefined configuration for use case
    
    Args:
        use_case: One of 'cancer_detection', 'medical_imaging', 'pandemic_forecasting'
        
    Returns:
        ExperimentConfig object
    """
    if use_case not in PREDEFINED_CONFIGS:
        raise ValueError(f"Unknown use case: {use_case}. "
                        f"Available: {list(PREDEFINED_CONFIGS.keys())}")
    
    config_dict = PREDEFINED_CONFIGS[use_case]
    return ExperimentConfig(**config_dict)


def create_output_directories(config: ExperimentConfig):
    """
    Create output directories for experiment
    
    Args:
        config: Experiment configuration
    """
    dirs = [
        config.output_dir,
        os.path.join(config.output_dir, 'models'),
        os.path.join(config.output_dir, 'figures'),
        os.path.join(config.output_dir, 'tables'),
        os.path.join(config.output_dir, 'logs')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"✓ Created output directories in {config.output_dir}")


if __name__ == "__main__":
    print("="*70)
    print("Testing Configuration Manager")
    print("="*70)
    
    # Create config manager
    config_mgr = ConfigManager()
    
    # Update configuration
    config_mgr.update(
        name="test_experiment",
        num_rounds=100,
        epsilon=1.5
    )
    
    # Print configuration
    config_mgr.print_config()
    
    # Save configuration
    config_mgr.save("test_config.json")
    
    # Load predefined config
    print("\nLoading predefined config for cancer detection:")
    cancer_config = get_config('cancer_detection')
    print(f"  Model: {cancer_config.model_name}")
    print(f"  Rounds: {cancer_config.num_rounds}")
    print(f"  Epsilon: {cancer_config.epsilon}")
    
    print("\n" + "="*70)
    print("Configuration tests passed!")
    print("="*70)
