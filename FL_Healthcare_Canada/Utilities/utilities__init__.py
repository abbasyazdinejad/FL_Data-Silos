"""
Sample of Utilities module for FL Healthcare experiments
Provides:
- Configuration management
- Logging and monitoring
- Reproducibility tools
"""

from .config import (
    ExperimentConfig,
    ConfigManager,
    get_config,
    create_output_directories,
    PREDEFINED_CONFIGS
)

from .logger import (
    ExperimentLogger,
    create_logger,
    SimpleLogger
)

from .reproducibility import (
    set_seed,
    set_deterministic,
    make_reproducible,
    get_environment_info,
    save_environment_info,
    check_reproducibility_requirements,
    create_checkpoint,
    load_checkpoint,
    verify_reproducibility,
    ReproducibilityManager
)

__all__ = [
    # Configuration
    'ExperimentConfig',
    'ConfigManager',
    'get_config',
    'create_output_directories',
    'PREDEFINED_CONFIGS',
    
    # Logging
    'ExperimentLogger',
    'create_logger',
    'SimpleLogger',
    
    # Reproducibility
    'set_seed',
    'set_deterministic',
    'make_reproducible',
    'get_environment_info',
    'save_environment_info',
    'check_reproducibility_requirements',
    'create_checkpoint',
    'load_checkpoint',
    'verify_reproducibility',
    'ReproducibilityManager'
]

__version__ = '1.0.0'
__author__ = 'Abbas Yazdinejad and Jude Kong'


def setup_experiment(
    experiment_name: str,
    use_case: str = 'cancer_detection',
    seed: int = 42,
    deterministic: bool = True,
    log_dir: str = './logs',
    output_dir: str = './results'
) -> tuple:
    """
    Complete experiment setup (convenience function)
    
    Sets up:
    - Configuration
    - Logging
    - Reproducibility
    - Output directories
    
    Args:
        experiment_name: Name of experiment
        use_case: Use case name
        seed: Random seed
        deterministic: Enable deterministic mode
        log_dir: Log directory
        output_dir: Output directory
        
    Returns:
        Tuple of (config, logger, repro_manager)
    """
    # Configuration
    config_mgr = ConfigManager()
    config_mgr.config = get_config(use_case)
    config_mgr.config.name = experiment_name
    config_mgr.config.seed = seed
    config_mgr.config.output_dir = output_dir
    
    # Create directories
    create_output_directories(config_mgr.config)
    
    # Logger
    logger = create_logger(experiment_name, log_dir=log_dir, verbose=True)
    logger.log_experiment_start()
    logger.log_config(config_mgr.to_dict())
    
    # Reproducibility
    repro_manager = ReproducibilityManager(
        seed=seed,
        deterministic=deterministic,
        save_dir=output_dir
    )
    repro_manager.setup()
    
    return config_mgr, logger, repro_manager


# Quick setup for common use cases
def quick_setup_cancer_detection(seed: int = 42):
    """Quick setup for cancer detection experiment"""
    return setup_experiment(
        experiment_name='cancer_detection',
        use_case='cancer_detection',
        seed=seed
    )


def quick_setup_pathmnist(seed: int = 42):
    """Quick setup for PathMNIST experiment"""
    return setup_experiment(
        experiment_name='pathmnist',
        use_case='medical_imaging',
        seed=seed
    )


if __name__ == "__main__":
    print("="*70)
    print("Testing Utilities Module")
    print("="*70)
    
    # Test quick setup
    print("\nQuick setup for cancer detection:")
    config_mgr, logger, repro_mgr = quick_setup_cancer_detection(seed=42)
    
    print("\n✓ Experiment setup complete!")
    print(f"  Config: {config_mgr.config.name}")
    print(f"  Logger: {logger.experiment_name}")
    print(f"  Seed: {repro_mgr.seed}")
    
    # Test logging
    logger.log_round(1, {'loss': 0.5, 'accuracy': 0.85})
    logger.log_round(2, {'loss': 0.4, 'accuracy': 0.88})
    
    logger.save_metrics()
    logger.log_experiment_end(total_time=120.0)
    
    print("\n" + "="*70)
    print("Utilities tests passed!")
    print("="*70)
