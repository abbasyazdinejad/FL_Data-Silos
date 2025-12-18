"""
Demo Federated data loader for synthetic provincial datasets
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Label vector (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FederatedDataLoader:
    """
    Federated data loader for provincial healthcare datasets
    """
    
    def __init__(
        self,
        datasets: Dict[str, pd.DataFrame],
        batch_size: int = 32,
        test_split: float = 0.2,
        normalize: bool = True,
        seed: int = 42
    ):
        """
        Initialize federated data loader
        
        Args:
            datasets: Dict mapping province/client names to DataFrames
            batch_size: Batch size for training
            test_split: Fraction of data for testing
            normalize: Whether to normalize features
            seed: Random seed
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.test_split = test_split
        self.normalize = normalize
        self.seed = seed
        
        # Store client names
        self.client_names = list(datasets.keys())
        
        # Prepare data
        self.client_loaders = {}
        self.test_loaders = {}
        self.scalers = {}
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare train/test splits and loaders for all clients"""
        from sklearn.model_selection import train_test_split
        
        for client_name, df in self.datasets.items():
            # Separate features and labels
            feature_cols = [col for col in df.columns 
                          if col not in ['diagnosis', 'province', 'phenotype', 
                                        'institution', 'icu_overload', 'Province', 'Cancer']]
            
            # Determine label column
            if 'diagnosis' in df.columns:
                label_col = 'diagnosis'
            elif 'Cancer' in df.columns:
                label_col = 'Cancer'
            elif 'phenotype' in df.columns:
                label_col = 'phenotype'
            elif 'icu_overload' in df.columns:
                label_col = 'icu_overload'
            else:
                raise ValueError(f"No label column found in {client_name} dataset")
            
            X = df[feature_cols].values
            y = df[label_col].values
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_split, random_state=self.seed,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Normalize if requested
            if self.normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                self.scalers[client_name] = scaler
            
            # Create PyTorch datasets
            train_dataset = TabularDataset(X_train, y_train)
            test_dataset = TabularDataset(X_test, y_test)
            
            # Create data loaders
            self.client_loaders[client_name] = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            self.test_loaders[client_name] = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            print(f"{client_name.capitalize()}: "
                  f"Train={len(train_dataset)}, Test={len(test_dataset)}, "
                  f"Positive rate={y_train.mean():.3f}")
    
    def get_client_loader(self, client_name: str) -> DataLoader:
        """Get training DataLoader for a client"""
        return self.client_loaders[client_name]
    
    def get_test_loader(self, client_name: str) -> DataLoader:
        """Get test DataLoader for a client"""
        return self.test_loaders[client_name]
    
    def get_global_test_loader(self) -> DataLoader:
        """Get combined test DataLoader from all clients"""
        # Combine all test data
        all_features = []
        all_labels = []
        
        for test_loader in self.test_loaders.values():
            for features, labels in test_loader:
                all_features.append(features)
                all_labels.append(labels)
        
        # Concatenate
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Create dataset and loader
        dataset = TabularDataset(features.numpy(), labels.numpy())
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return loader
    
    def get_num_clients(self) -> int:
        """Get number of federated clients"""
        return len(self.client_names)
    
    def get_feature_dim(self) -> int:
        """Get feature dimensionality"""
        # Get first batch from first client
        first_client = self.client_names[0]
        first_batch = next(iter(self.client_loaders[first_client]))
        return first_batch[0].shape[1]
    
    def get_num_classes(self) -> int:
        """Get number of output classes"""
        # Get labels from first client
        first_client = self.client_names[0]
        all_labels = []
        for _, labels in self.test_loaders[first_client]:
            all_labels.extend(labels.numpy())
        return len(np.unique(all_labels))
    
    def get_client_sample_counts(self) -> Dict[str, int]:
        """Get number of samples per client"""
        counts = {}
        for client_name, loader in self.client_loaders.items():
            counts[client_name] = len(loader.dataset)
        return counts


def load_from_csv_files(
    file_paths: Dict[str, str],
    batch_size: int = 32,
    test_split: float = 0.2,
    normalize: bool = True,
    seed: int = 42
) -> FederatedDataLoader:
    """
    Load datasets from CSV files and create federated data loader
    
    Args:
        file_paths: Dict mapping client names to CSV file paths
        batch_size: Batch size
        test_split: Test split fraction
        normalize: Whether to normalize features
        seed: Random seed
        
    Returns:
        FederatedDataLoader instance
    """
    datasets = {}
    for client_name, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        datasets[client_name] = df
        print(f"Loaded {client_name} from {file_path}: {len(df)} samples")
    
    return FederatedDataLoader(
        datasets=datasets,
        batch_size=batch_size,
        test_split=test_split,
        normalize=normalize,
        seed=seed
    )


if __name__ == "__main__":
    # Example usage
    print("Example: Loading provincial cancer datasets")
    
    # Simulate loading from CSV files
    file_paths = {
        'ontario': './data/ontario_cancer_data.csv',
        'alberta': './data/alberta_cancer_data.csv',
        'quebec': './data/quebec_cancer_data.csv'
    }
    
    # For demonstration, create synthetic data
    from synthetic_generator import generate_cancer_datasets
    datasets = generate_cancer_datasets(seed=42)
    
    # Create federated data loader
    fed_loader = FederatedDataLoader(
        datasets=datasets,
        batch_size=32,
        test_split=0.2,
        normalize=True,
        seed=42
    )
    
    print(f"\nNumber of clients: {fed_loader.get_num_clients()}")
    print(f"Feature dimension: {fed_loader.get_feature_dim()}")
    print(f"Number of classes: {fed_loader.get_num_classes()}")
    
    # Test loading a batch
    ontario_loader = fed_loader.get_client_loader('ontario')
    batch_features, batch_labels = next(iter(ontario_loader))
    print(f"\nBatch shape: {batch_features.shape}")
    print(f"Label shape: {batch_labels.shape}")
