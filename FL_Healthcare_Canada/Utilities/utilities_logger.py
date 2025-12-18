"""
Demo of Logging utilities for FL Healthcare experiments
Provides structured logging for:
- Training progress
- Evaluation metrics
- Privacy accounting
- Governance events
- File output (logs, results)
"""

import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import json


class ExperimentLogger:
    """
    Comprehensive logger for FL experiments
    
    Logs to:
    - Console (formatted output)
    - File (detailed logs)
    - JSON (structured metrics)
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./logs",
        log_level: int = logging.INFO,
        log_to_file: bool = True,
        log_to_console: bool = True
    ):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of experiment
            log_dir: Directory for log files
            log_level: Logging level
            log_to_file: Enable file logging
            log_to_console: Enable console logging
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.log_level = log_level
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # File handler
        if log_to_file:
            log_file = os.path.join(log_dir, f"{experiment_name}_{self.timestamp}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics_history = []
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_{self.timestamp}_metrics.json")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log experiment configuration
        
        Args:
            config: Configuration dictionary
        """
        self.info("="*70)
        self.info("EXPERIMENT CONFIGURATION")
        self.info("="*70)
        for key, value in config.items():
            self.info(f"  {key}: {value}")
        self.info("="*70)
    
    def log_round(
        self,
        round_num: int,
        metrics: Dict[str, float],
        prefix: str = ""
    ):
        """
        Log federated round metrics
        
        Args:
            round_num: Round number
            metrics: Dict of metrics
            prefix: Optional prefix for message
        """
        msg = f"{prefix}Round {round_num:3d}"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}={value:.4f}"
            else:
                msg += f" | {key}={value}"
        
        self.info(msg)
        
        # Store metrics
        self.metrics_history.append({
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None
    ):
        """
        Log training epoch
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_acc: Training accuracy (optional)
            val_loss: Validation loss (optional)
            val_acc: Validation accuracy (optional)
        """
        msg = f"Epoch {epoch:3d} | Loss: {train_loss:.4f}"
        
        if train_acc is not None:
            msg += f" | Acc: {train_acc:.4f}"
        
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
        
        if val_acc is not None:
            msg += f" | Val Acc: {val_acc:.4f}"
        
        self.info(msg)
    
    def log_evaluation(
        self,
        metrics: Dict[str, float],
        dataset: str = "test"
    ):
        """
        Log evaluation results
        
        Args:
            metrics: Evaluation metrics
            dataset: Dataset name
        """
        self.info(f"\nEvaluation on {dataset} set:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
    
    def log_privacy(self, privacy_spent: Dict[str, float]):
        """
        Log privacy budget
        
        Args:
            privacy_spent: Privacy statistics
        """
        self.info("\nPrivacy Budget:")
        self.info(f"  Epsilon (ε): {privacy_spent['epsilon']:.4f}")
        self.info(f"  Delta (δ): {privacy_spent['delta']:.2e}")
        self.info(f"  Budget remaining: {privacy_spent.get('budget_remaining', 'N/A')}")
    
    def log_governance(self, governance_stats: Dict[str, Any]):
        """
        Log governance statistics
        
        Args:
            governance_stats: Governance metrics
        """
        self.info("\nGovernance Statistics:")
        for component, stats in governance_stats.items():
            self.info(f"  {component}:")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    self.info(f"    {key}: {value}")
            else:
                self.info(f"    {stats}")
    
    def log_experiment_start(self):
        """Log experiment start"""
        self.info("\n" + "="*70)
        self.info(f"STARTING EXPERIMENT: {self.experiment_name}")
        self.info(f"Timestamp: {self.timestamp}")
        self.info("="*70)
    
    def log_experiment_end(self, total_time: float):
        """
        Log experiment end
        
        Args:
            total_time: Total experiment time in seconds
        """
        self.info("\n" + "="*70)
        self.info(f"EXPERIMENT COMPLETED: {self.experiment_name}")
        self.info(f"Total time: {total_time/60:.2f} minutes")
        self.info("="*70)
    
    def save_metrics(self):
        """Save metrics history to JSON file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.info(f"\n✓ Metrics saved to {self.metrics_file}")
    
    def load_metrics(self, metrics_file: str):
        """
        Load metrics from file
        
        Args:
            metrics_file: Path to metrics JSON file
        """
        with open(metrics_file, 'r') as f:
            self.metrics_history = json.load(f)
        
        self.info(f"✓ Loaded {len(self.metrics_history)} metric entries")
    
    def get_metrics(self) -> list:
        """Get metrics history"""
        return self.metrics_history


def create_logger(
    experiment_name: str,
    log_dir: str = "./logs",
    verbose: bool = True
) -> ExperimentLogger:
    """
    Create experiment logger (convenience function)
    
    Args:
        experiment_name: Experiment name
        log_dir: Log directory
        verbose: Enable console output
        
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        log_level=logging.INFO if verbose else logging.WARNING,
        log_to_file=True,
        log_to_console=verbose
    )


# Simple print-based logger for quick use
class SimpleLogger:
    """Lightweight logger using print statements"""
    
    def __init__(self, name: str = "Experiment"):
        self.name = name
    
    def info(self, msg: str):
        print(f"[INFO] {msg}")
    
    def warning(self, msg: str):
        print(f"[WARNING] {msg}")
    
    def error(self, msg: str):
        print(f"[ERROR] {msg}")
    
    def log_round(self, round_num: int, loss: float, acc: float = None):
        msg = f"Round {round_num:3d}: Loss={loss:.4f}"
        if acc is not None:
            msg += f", Acc={acc:.4f}"
        self.info(msg)


if __name__ == "__main__":
    print("="*70)
    print("Testing Experiment Logger")
    print("="*70)
    
    # Create logger
    logger = create_logger("test_experiment", log_dir="./test_logs")
    
    # Log experiment start
    logger.log_experiment_start()
    
    # Log configuration
    config = {
        'num_rounds': 50,
        'learning_rate': 0.001,
        'epsilon': 1.0
    }
    logger.log_config(config)
    
    # Log some rounds
    print("\nLogging training rounds:")
    for round_num in range(1, 6):
        metrics = {
            'loss': 0.5 - round_num * 0.05,
            'accuracy': 0.7 + round_num * 0.03
        }
        logger.log_round(round_num, metrics)
    
    # Log evaluation
    eval_metrics = {
        'accuracy': 0.92,
        'auc': 0.95,
        'f1': 0.91
    }
    logger.log_evaluation(eval_metrics)
    
    # Log privacy
    privacy_spent = {
        'epsilon': 1.0,
        'delta': 1e-5,
        'budget_remaining': 1.0
    }
    logger.log_privacy(privacy_spent)
    
    # Save metrics
    logger.save_metrics()
    
    # Log experiment end
    logger.log_experiment_end(total_time=120.5)
    
    print("\n" + "="*70)
    print("Logger tests passed!")
    print("="*70)
