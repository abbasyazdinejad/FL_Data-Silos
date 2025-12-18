"""
Example of Federated learning client implementation

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import copy


class FederatedClient:
    """
    Generic federated learning client
    
    Each client maintains:
    - Local model copy
    - Local dataset
    - Training configuration
    - Update history
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        local_epochs: int = 5,
        optimizer_name: str = 'adam'
    ):
        """
        Initialize federated client
        
        Args:
            client_id: Unique client identifier (e.g., 'ontario', 'hospital_1')
            model: PyTorch model (architecture only, parameters will be synchronized)
            train_loader: Training data loader for this client
            device: Computing device (cpu, cuda, mps)
            learning_rate: Learning rate for local training
            local_epochs: Number of local training epochs per round
            optimizer_name: Optimizer type ('adam', 'sgd', 'adamw')
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer_name)
        
        # Loss function (can be customized)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'rounds': [],
            'losses': [],
            'samples_trained': []
        }
    
    def _create_optimizer(self, optimizer_name: str):
        """Create optimizer based on name"""
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get current model parameters
        
        Returns:
            Dict mapping parameter names to tensors
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set model parameters from global model
        
        Args:
            parameters: Dict of parameter tensors from server
        """
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data = parameters[name].to(self.device).clone()
    
    def train_local(
        self, 
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
        round_num: Optional[int] = None
    ) -> Dict:
        """
        Perform local training for specified number of epochs
        
        This is the core local training procedure:
        1. Synchronize with global model (if provided)
        2. Train on local data for local_epochs
        3. Return updated parameters and statistics
        
        Args:
            global_parameters: Global model parameters to start from (if None, uses current)
            round_num: Current federated round number (for tracking)
            
        Returns:
            Dict containing:
                - parameters: Updated model parameters
                - num_samples: Number of samples trained on
                - loss: Average loss over all local epochs
                - client_id: This client's ID
        """
        # Synchronize with global model if provided
        if global_parameters is not None:
            self.set_model_parameters(global_parameters)
        
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        num_batches = 0
        
        # Local training loop
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                batch_loss = loss.item()
                batch_size = len(data)
                
                epoch_loss += batch_loss * batch_size
                num_samples += batch_size
                num_batches += 1
            
            total_loss += epoch_loss
        
        # Compute average loss
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        # Update history
        if round_num is not None:
            self.history['rounds'].append(round_num)
            self.history['losses'].append(avg_loss)
            self.history['samples_trained'].append(num_samples)
        
        return {
            'parameters': self.get_model_parameters(),
            'num_samples': num_samples // self.local_epochs,  # Samples per epoch
            'loss': avg_loss,
            'client_id': self.client_id
        }
    
    def evaluate(
        self, 
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            test_loader: Test data loader
            return_predictions: Whether to return predictions and labels
            
        Returns:
            Dict of evaluation metrics
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item() * len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
                
                if return_predictions:
                    all_predictions.extend(pred.cpu().numpy().flatten())
                    all_labels.extend(target.cpu().numpy())
        
        result = {
            'loss': test_loss / total if total > 0 else 0.0,
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total,
            'client_id': self.client_id
        }
        
        if return_predictions:
            result['predictions'] = all_predictions
            result['labels'] = all_labels
        
        return result
    
    def get_dataset_size(self) -> int:
        """Get size of local training dataset"""
        return len(self.train_loader.dataset)
    
    def reset_optimizer(self):
        """Reset optimizer state (useful between rounds)"""
        optimizer_class = type(self.optimizer)
        self.optimizer = optimizer_class(
            self.model.parameters(), 
            lr=self.learning_rate
        )
    
    def __repr__(self):
        """String representation"""
        return (f"FederatedClient(id='{self.client_id}', "
                f"dataset_size={self.get_dataset_size()}, "
                f"local_epochs={self.local_epochs})")


class FedProxClient(FederatedClient):
    """
    FedProx client with proximal term
    
    FedProx adds a proximal term to the loss function to handle
    heterogeneous data and systems. See Li et al. 2020.
    
    Loss = CrossEntropy + (mu/2) * ||theta - theta_global||^2
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        local_epochs: int = 5,
        mu: float = 0.01,
        optimizer_name: str = 'sgd'
    ):
        """
        Initialize FedProx client
        
        Args:
            mu: Proximal term coefficient (controls regularization strength)
            Other args: Same as FederatedClient
        """
        super().__init__(
            client_id, model, train_loader, device,
            learning_rate, local_epochs, optimizer_name
        )
        self.mu = mu
        self.global_parameters = None
    
    def train_local(
        self,
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
        round_num: Optional[int] = None
    ) -> Dict:
        """
        Local training with proximal term
        
        Args:
            global_parameters: Global model parameters
            round_num: Current round number
            
        Returns:
            Training results dict
        """
        # Store global parameters for proximal term
        if global_parameters is not None:
            self.set_model_parameters(global_parameters)
            self.global_parameters = {
                name: param.clone().detach() 
                for name, param in global_parameters.items()
            }
        
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for epoch in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                ce_loss = self.criterion(output, target)
                
                # Add proximal term
                if self.global_parameters is not None:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if name in self.global_parameters:
                            prox_loss += ((param - self.global_parameters[name]) ** 2).sum()
                    
                    loss = ce_loss + (self.mu / 2.0) * prox_loss
                else:
                    loss = ce_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(data)
                num_samples += len(data)
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        return {
            'parameters': self.get_model_parameters(),
            'num_samples': num_samples // self.local_epochs,
            'loss': avg_loss,
            'client_id': self.client_id
        }


if __name__ == "__main__":
    print("="*60)
    print("Testing Federated Client")
    print("="*60)
    
    # Create dummy model and data
    from torch.utils.data import TensorDataset
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = torch.device('cpu')
    
    # Test FederatedClient
    print("\n1. FederatedClient")
    print("-"*60)
    client = FederatedClient(
        client_id='test_client',
        model=model,
        train_loader=train_loader,
        device=device,
        learning_rate=0.01,
        local_epochs=2
    )
    
    print(f"Client: {client}")
    print(f"Dataset size: {client.get_dataset_size()}")
    
    # Test local training
    print("\n2. Local Training")
    print("-"*60)
    result = client.train_local(round_num=1)
    print(f"Loss: {result['loss']:.4f}")
    print(f"Samples: {result['num_samples']}")
    print(f"Client ID: {result['client_id']}")
    
    # Test evaluation
    print("\n3. Evaluation")
    print("-"*60)
    eval_result = client.evaluate(train_loader)
    print(f"Test loss: {eval_result['loss']:.4f}")
    print(f"Accuracy: {eval_result['accuracy']:.4f}")
    
    # Test FedProxClient
    print("\n4. FedProxClient")
    print("-"*60)
    fedprox_client = FedProxClient(
        client_id='fedprox_client',
        model=copy.deepcopy(model),
        train_loader=train_loader,
        device=device,
        mu=0.01
    )
    
    result = fedprox_client.train_local(round_num=1)
    print(f"FedProx loss: {result['loss']:.4f}")
    
    print("\n" + "="*60)
    print("All client tests passed!")
    print("="*60)
