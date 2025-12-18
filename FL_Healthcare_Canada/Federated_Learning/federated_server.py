"""
Federated learning server implementation

Coordinates federated learning across multiple clients:
- Manages global model
- Orchestrates federated rounds
- Aggregates client updates
- Tracks training progress
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import copy


class FederatedServer:
    """
    Federated learning server/coordinator
    
    The server:
    1. Maintains the global model
    2. Distributes global model to clients
    3. Receives and aggregates client updates
    4. Tracks training progress
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        device: torch.device,
        aggregation_method: str = 'weighted_avg'
    ):
        """
        Initialize federated server
        
        Args:
            global_model: Global model architecture
            device: Computing device
            aggregation_method: Method for aggregating client models
                ('weighted_avg', 'mean', 'median', 'secure')
        """
        self.global_model = global_model.to(device)
        self.device = device
        self.aggregation_method = aggregation_method
        
        # Training history
        self.history = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'num_clients': [],
            'total_samples': []
        }
        
        # Current round
        self.current_round = 0
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters"""
        return {name: param.data.clone() for name, param in self.global_model.named_parameters()}
    
    def set_global_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set global model parameters"""
        for name, param in self.global_model.named_parameters():
            if name in parameters:
                param.data = parameters[name].to(self.device).clone()
    
    def aggregate_client_updates(
        self,
        client_updates: List[Dict],
        use_weights: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate updates from multiple clients
        
        Args:
            client_updates: List of dicts containing 'parameters' and 'num_samples'
            use_weights: Whether to weight by number of samples
            
        Returns:
            Aggregated model parameters
        """
        from aggregation import aggregate_models
        
        # Extract parameters and weights
        client_params = [update['parameters'] for update in client_updates]
        
        if use_weights:
            client_weights = [update['num_samples'] for update in client_updates]
        else:
            client_weights = None
        
        # Aggregate using specified method
        aggregated_params = aggregate_models(
            client_params,
            aggregation_method=self.aggregation_method,
            client_weights=client_weights
        )
        
        return aggregated_params
    
    def federated_round(
        self,
        clients: List,
        round_num: int,
        evaluate: bool = False,
        test_loader = None
    ) -> Dict:
        """
        Execute one round of federated learning
        
        Standard FL round:
        1. Send global model to clients
        2. Clients train locally
        3. Receive client updates
        4. Aggregate updates
        5. Update global model
        
        Args:
            clients: List of FederatedClient objects
            round_num: Current round number
            evaluate: Whether to evaluate after aggregation
            test_loader: Test data loader (if evaluate=True)
            
        Returns:
            Dict of round statistics
        """
        # Get current global parameters
        global_params = self.get_global_parameters()
        
        # Collect updates from all clients
        client_updates = []
        total_loss = 0.0
        total_samples = 0
        
        for client in clients:
            # Client performs local training
            update = client.train_local(global_params, round_num)
            client_updates.append(update)
            
            # Track statistics
            total_loss += update['loss'] * update['num_samples']
            total_samples += update['num_samples']
        
        # Aggregate client updates
        aggregated_params = self.aggregate_client_updates(client_updates)
        
        # Update global model
        self.set_global_parameters(aggregated_params)
        
        # Compute average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Evaluate if requested
        accuracy = None
        if evaluate and test_loader is not None:
            eval_result = self.evaluate_global(test_loader)
            accuracy = eval_result['accuracy']
        
        # Update history
        self.history['round'].append(round_num)
        self.history['loss'].append(avg_loss)
        if accuracy is not None:
            self.history['accuracy'].append(accuracy)
        self.history['num_clients'].append(len(clients))
        self.history['total_samples'].append(total_samples)
        
        self.current_round = round_num
        
        return {
            'round': round_num,
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_clients': len(clients),
            'total_samples': total_samples
        }
    
    def evaluate_global(self, test_loader) -> Dict[str, float]:
        """
        Evaluate global model on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item() * len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        return {
            'loss': test_loss / total if total > 0 else 0.0,
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total
        }
    
    def train(
        self,
        clients: List,
        num_rounds: int,
        evaluate_every: int = 1,
        test_loader = None,
        verbose: bool = True
    ) -> Dict:
        """
        Complete federated training loop
        
        Args:
            clients: List of federated clients
            num_rounds: Number of federated rounds
            evaluate_every: Evaluate every N rounds
            test_loader: Test data loader for evaluation
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        if verbose:
            print(f"\nStarting federated training with {len(clients)} clients")
            print(f"Total rounds: {num_rounds}")
            print("-"*60)
        
        for round_num in range(1, num_rounds + 1):
            # Evaluate flag
            should_evaluate = (test_loader is not None and 
                             round_num % evaluate_every == 0)
            
            # Execute round
            round_stats = self.federated_round(
                clients, round_num,
                evaluate=should_evaluate,
                test_loader=test_loader
            )
            
            # Print progress
            if verbose:
                if should_evaluate:
                    print(f"Round {round_num}/{num_rounds}: "
                          f"Loss={round_stats['loss']:.4f}, "
                          f"Acc={round_stats['accuracy']:.4f}")
                else:
                    print(f"Round {round_num}/{num_rounds}: "
                          f"Loss={round_stats['loss']:.4f}")
        
        if verbose:
            print("-"*60)
            print("Training completed!")
        
        return self.history
    
    def save_model(self, path: str):
        """
        Save global model to file
        
        Args:
            path: File path
        """
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'round': self.current_round,
            'history': self.history
        }, path)
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load global model from file
        
        Args:
            path: File path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.current_round = checkpoint.get('round', 0)
        self.history = checkpoint.get('history', self.history)
    
    def __repr__(self):
        """String representation"""
        return (f"FederatedServer(aggregation='{self.aggregation_method}', "
                f"current_round={self.current_round})")


if __name__ == "__main__":
    print("="*60)
    print("Testing Federated Server")
    print("="*60)
    
    # Create dummy setup
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    device = torch.device('cpu')
    
    # Create server
    print("\n1. Initialize Server")
    print("-"*60)
    server = FederatedServer(model, device, aggregation_method='weighted_avg')
    print(f"Server: {server}")
    
    # Test parameter get/set
    print("\n2. Test Parameter Operations")
    print("-"*60)
    params = server.get_global_parameters()
    print(f"Number of parameter tensors: {len(params)}")
    
    # Create dummy clients
    print("\n3. Create Dummy Clients")
    print("-"*60)
    clients = []
    for i in range(3):
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)
        
        # Import client (would be from client.py in real code)
        from torch.utils.data import DataLoader as DL
        
        class DummyClient:
            def __init__(self, cid, loader):
                self.client_id = cid
                self.loader = loader
            
            def train_local(self, global_params, round_num):
                return {
                    'parameters': global_params,
                    'num_samples': 50,
                    'loss': 0.5 + i * 0.1,
                    'client_id': self.client_id
                }
        
        clients.append(DummyClient(f'client_{i}', loader))
        print(f"  Client {i}: 50 samples")
    
    # Test federated round
    print("\n4. Execute Federated Round")
    print("-"*60)
    result = server.federated_round(clients, round_num=1)
    print(f"Round: {result['round']}")
    print(f"Loss: {result['loss']:.4f}")
    print(f"Clients: {result['num_clients']}")
    print(f"Total samples: {result['total_samples']}")
    
    print("\n" + "="*60)
    print("Server tests completed!")
    print("="*60)
