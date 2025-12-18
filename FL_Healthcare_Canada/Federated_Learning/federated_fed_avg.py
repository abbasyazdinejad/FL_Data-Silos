"""
Example of FedAvg (Federated Averaging) implementation

The original and most widely-used federated learning algorithm.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from torch.utils.data import DataLoader


class FedAvgTrainer:
    """
    High-level FedAvg training orchestrator
    
    Simplifies federated learning by providing a clean API:
    1. Create trainer with model and clients
    2. Call train() method
    3. Get trained model and history
    
    Example:
        >>> trainer = FedAvgTrainer(model, clients, device)
        >>> history = trainer.train(num_rounds=50)
        >>> trained_model = trainer.get_global_model()
    """
    
    def __init__(
        self,
        model: nn.Module,
        clients: List,
        device: torch.device,
        aggregation_method: str = 'weighted_avg',
        test_loader: Optional[DataLoader] = None
    ):
        """
        Initialize FedAvg trainer
        
        Args:
            model: Global model architecture
            clients: List of FederatedClient objects
            device: Computing device
            aggregation_method: Aggregation method ('weighted_avg', 'mean', 'median')
            test_loader: Optional global test loader for evaluation
        """
        from .server import FederatedServer
        
        self.clients = clients
        self.device = device
        self.test_loader = test_loader
        
        # Create federated server
        self.server = FederatedServer(
            global_model=model,
            device=device,
            aggregation_method=aggregation_method
        )
        
        # Training history
        self.history = {
            'rounds': [],
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': [],
            'client_losses': []
        }
    
    def train(
        self,
        num_rounds: int,
        evaluate_every: int = 1,
        verbose: bool = True,
        save_checkpoints: bool = False,
        checkpoint_dir: str = './checkpoints'
    ) -> Dict:
        """
        Execute FedAvg training
        
        Args:
            num_rounds: Number of federated rounds
            evaluate_every: Evaluate every N rounds
            verbose: Print progress
            save_checkpoints: Save model checkpoints
            checkpoint_dir: Directory for checkpoints
            
        Returns:
            Training history
        """
        import os
        
        if verbose:
            print("\n" + "="*70)
            print("FEDERATED AVERAGING (FedAvg) TRAINING")
            print("="*70)
            print(f"Clients: {len(self.clients)}")
            print(f"Rounds: {num_rounds}")
            print(f"Device: {self.device}")
            print("-"*70)
        
        for round_num in range(1, num_rounds + 1):
            # Execute federated round
            should_evaluate = (self.test_loader is not None and 
                             round_num % evaluate_every == 0)
            
            round_stats = self.server.federated_round(
                clients=self.clients,
                round_num=round_num,
                evaluate=should_evaluate,
                test_loader=self.test_loader
            )
            
            # Record history
            self.history['rounds'].append(round_num)
            self.history['train_loss'].append(round_stats['loss'])
            
            if should_evaluate:
                self.history['test_loss'].append(round_stats.get('test_loss'))
                self.history['test_accuracy'].append(round_stats.get('accuracy'))
            
            # Print progress
            if verbose:
                if should_evaluate:
                    print(f"Round {round_num:3d}/{num_rounds}: "
                          f"Loss={round_stats['loss']:.4f}, "
                          f"Acc={round_stats['accuracy']:.4f}")
                else:
                    print(f"Round {round_num:3d}/{num_rounds}: "
                          f"Loss={round_stats['loss']:.4f}")
            
            # Save checkpoint
            if save_checkpoints and round_num % 10 == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{checkpoint_dir}/fedavg_round_{round_num}.pt"
                self.server.save_model(checkpoint_path)
                if verbose:
                    print(f"  └─ Saved checkpoint: {checkpoint_path}")
        
        if verbose:
            print("-"*70)
            print("Training completed!")
            print("="*70)
        
        return self.history
    
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict:
        """
        Evaluate global model
        
        Args:
            test_loader: Test data loader (uses self.test_loader if None)
            
        Returns:
            Evaluation metrics
        """
        if test_loader is None:
            test_loader = self.test_loader
        
        if test_loader is None:
            raise ValueError("No test loader provided")
        
        return self.server.evaluate_global(test_loader)
    
    def get_global_model(self) -> nn.Module:
        """Get the trained global model"""
        return self.server.global_model
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get global model parameters"""
        return self.server.get_global_parameters()
    
    def save_model(self, path: str):
        """Save global model"""
        self.server.save_model(path)
    
    def load_model(self, path: str):
        """Load global model"""
        self.server.load_model(path)
    
    def get_client_statistics(self) -> Dict:
        """
        Get statistics about clients
        
        Returns:
            Dict with client information
        """
        stats = {
            'num_clients': len(self.clients),
            'client_ids': [c.client_id for c in self.clients],
            'dataset_sizes': {c.client_id: c.get_dataset_size() for c in self.clients},
            'total_samples': sum(c.get_dataset_size() for c in self.clients)
        }
        
        return stats


if __name__ == "__main__":
    print("="*70)
    print("Testing FedAvg Trainer")
    print("="*70)
    
    # This would normally import from other modules
    # For testing, we'll create a minimal setup
    
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Create dummy clients (would import FederatedClient in real usage)
    class DummyClient:
        def __init__(self, client_id, data_size):
            self.client_id = client_id
            self.data_size = data_size
            self.model = None
        
        def get_dataset_size(self):
            return self.data_size
        
        def train_local(self, global_params, round_num):
            # Simulate training
            return {
                'parameters': global_params,
                'num_samples': self.data_size,
                'loss': 0.5,
                'client_id': self.client_id
            }
    
    clients = [
        DummyClient('ontario', 1000),
        DummyClient('alberta', 500),
        DummyClient('quebec', 750)
    ]
    
    device = torch.device('cpu')
    
    # Create FedAvg trainer
    print("\nCreating FedAvg Trainer...")
    trainer = FedAvgTrainer(model, clients, device)
    
    print("\nClient Statistics:")
    stats = trainer.get_client_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("FedAvg Trainer ready for training!")
    print("="*70)
