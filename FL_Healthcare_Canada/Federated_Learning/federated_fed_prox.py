"""
Example of FedProx (Federated Proximal) algorithm implementation

Handles non-IID data and heterogeneous systems by adding a proximal term:
Loss = CrossEntropy + (mu/2) * ||theta - theta_global||^2

The proximal term keeps local models close to the global model,
improving convergence in heterogeneous settings.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from torch.utils.data import DataLoader


class FedProxTrainer:
    """
    High-level FedProx training orchestrator
    
    FedProx is particularly suitable for:
    - Non-IID data across clients (e.g., different provinces)
    - Heterogeneous compute resources
    - Partial client participation
    - Systems heterogeneity
    
    Example:
        >>> trainer = FedProxTrainer(model, clients, device, mu=0.01)
        >>> history = trainer.train(num_rounds=50)
        >>> trained_model = trainer.get_global_model()
    """
    
    def __init__(
        self,
        model: nn.Module,
        clients: List,
        device: torch.device,
        mu: float = 0.01,
        aggregation_method: str = 'weighted_avg',
        test_loader: Optional[DataLoader] = None
    ):
        """
        Initialize FedProx trainer
        
        Args:
            model: Global model architecture
            clients: List of FedProxClient objects
            device: Computing device
            mu: Proximal term coefficient (typically 0.001 to 0.1)
                - Higher mu: Stronger regularization, slower adaptation
                - Lower mu: Weaker regularization, closer to FedAvg
            aggregation_method: Aggregation method
            test_loader: Optional global test loader
        """
        from .server import FederatedServer
        
        self.clients = clients
        self.device = device
        self.mu = mu
        self.test_loader = test_loader
        
        # Ensure clients have mu parameter set
        for client in self.clients:
            if hasattr(client, 'mu'):
                client.mu = mu
        
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
            'proximal_terms': [],  # Track proximal regularization
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
        Execute FedProx training
        
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
            print("FEDERATED PROXIMAL (FedProx) TRAINING")
            print("="*70)
            print(f"Clients: {len(self.clients)}")
            print(f"Rounds: {num_rounds}")
            print(f"Proximal term (mu): {self.mu}")
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
                          f"Acc={round_stats['accuracy']:.4f}, "
                          f"mu={self.mu:.4f}")
                else:
                    print(f"Round {round_num:3d}/{num_rounds}: "
                          f"Loss={round_stats['loss']:.4f}")
            
            # Save checkpoint
            if save_checkpoints and round_num % 10 == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{checkpoint_dir}/fedprox_round_{round_num}.pt"
                self.server.save_model(checkpoint_path)
                if verbose:
                    print(f"  └─ Saved checkpoint: {checkpoint_path}")
        
        if verbose:
            print("-"*70)
            print("Training completed!")
            print(f"Final mu: {self.mu:.4f}")
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
        torch.save({
            'model_state_dict': self.server.global_model.state_dict(),
            'mu': self.mu,
            'history': self.history,
            'round': self.server.current_round
        }, path)
    
    def load_model(self, path: str):
        """Load global model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.server.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.mu = checkpoint.get('mu', self.mu)
        self.history = checkpoint.get('history', self.history)
        self.server.current_round = checkpoint.get('round', 0)
    
    def adaptive_mu(self, round_num: int, total_rounds: int, 
                    mu_start: float = 0.01, mu_end: float = 0.001):
        """
        Adaptive mu scheduling (decrease mu over time)
        
        Intuition: Start with strong regularization, relax as training progresses
        
        Args:
            round_num: Current round
            total_rounds: Total number of rounds
            mu_start: Starting mu value
            mu_end: Ending mu value
        """
        # Linear decay
        self.mu = mu_start + (mu_end - mu_start) * (round_num / total_rounds)
        
        # Update mu for all clients
        for client in self.clients:
            if hasattr(client, 'mu'):
                client.mu = self.mu
    
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
            'total_samples': sum(c.get_dataset_size() for c in self.clients),
            'mu': self.mu
        }
        
        return stats
    
    def compare_with_fedavg(self, convergence_threshold: float = 0.01) -> Dict:
        """
        Analysis: Compare FedProx behavior with FedAvg
        
        FedProx typically shows:
        - Better convergence in non-IID settings
        - More stable training
        - Slightly slower per-round updates (due to proximal term)
        
        Returns:
            Comparison metrics
        """
        return {
            'algorithm': 'FedProx',
            'mu': self.mu,
            'advantages': [
                'Handles non-IID data better',
                'More robust to stragglers',
                'Stable convergence'
            ],
            'compared_to': 'FedAvg (mu=0)',
            'recommended_for': 'Healthcare FL with heterogeneous provincial data'
        }


if __name__ == "__main__":
    print("="*70)
    print("Testing FedProx Trainer")
    print("="*70)
    
    import torch
    import torch.nn as nn
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Create dummy FedProx clients
    class DummyFedProxClient:
        def __init__(self, client_id, data_size, mu=0.01):
            self.client_id = client_id
            self.data_size = data_size
            self.mu = mu
        
        def get_dataset_size(self):
            return self.data_size
        
        def train_local(self, global_params, round_num):
            # Simulate training with proximal term
            return {
                'parameters': global_params,
                'num_samples': self.data_size,
                'loss': 0.5 + 0.01 * self.mu,  # Loss includes proximal term
                'client_id': self.client_id
            }
    
    clients = [
        DummyFedProxClient('ontario', 1000, mu=0.01),
        DummyFedProxClient('alberta', 500, mu=0.01),
        DummyFedProxClient('quebec', 750, mu=0.01)
    ]
    
    device = torch.device('cpu')
    
    # Create FedProx trainer
    print("\nCreating FedProx Trainer...")
    trainer = FedProxTrainer(model, clients, device, mu=0.01)
    
    print("\nClient Statistics:")
    stats = trainer.get_client_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nFedProx vs FedAvg:")
    comparison = trainer.compare_with_fedavg()
    print(f"  Algorithm: {comparison['algorithm']}")
    print(f"  Mu: {comparison['mu']}")
    print(f"  Advantages:")
    for adv in comparison['advantages']:
        print(f"    - {adv}")
    
    print("\n" + "="*70)
    print("FedProx Trainer ready for training!")
    print("="*70)
