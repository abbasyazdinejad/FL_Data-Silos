"""
PathMNIST dataset loader 
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np
from typing import List, Dict, Tuple


def load_pathmnist(
    split: str = 'train',
    download: bool = True,
    size: int = 28,
    as_rgb: bool = True
):
    """
    Load PathMNIST dataset
    
    Args:
        split: 'train', 'val', or 'test'
        download: Whether to download dataset
        size: Image size (28 for MNIST-like, or larger)
        as_rgb: Whether to use RGB (True) or grayscale (False)
        
    Returns:
        PathMNIST dataset
    """
    from medmnist import PathMNIST
    
    dataset = PathMNIST(
        split=split,
        download=download,
        size=size,
        as_rgb=as_rgb
    )
    
    return dataset


class PathMNISTFederatedLoader:
    """
    Federated data loader for PathMNIST across simulated hospitals
    """
    
    def __init__(
        self,
        num_clients: int = 5,
        batch_size: int = 32,
        download: bool = True,
        seed: int = 42,
        size: int = 28
    ):
        """
        Initialize federated PathMNIST loader
        
        Args:
            num_clients: Number of federated clients (hospitals)
            batch_size: Batch size for training
            download: Whether to download dataset
            seed: Random seed for reproducibility
            size: Image size (default 28 for MNIST-like)
        """
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.seed = seed
        self.size = size
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load datasets
        print("Loading PathMNIST dataset...")
        self.train_dataset = load_pathmnist('train', download=download, size=size)
        self.val_dataset = load_pathmnist('val', download=download, size=size)
        self.test_dataset = load_pathmnist('test', download=download, size=size)
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        
        # Wrap datasets for consistent format
        self.train_dataset = PathMNISTWrapper(self.train_dataset)
        self.val_dataset = PathMNISTWrapper(self.val_dataset)
        self.test_dataset = PathMNISTWrapper(self.test_dataset)
        
        # Split training data among clients
        self.client_indices = self._create_client_splits()
        
    def _create_client_splits(self) -> Dict[int, List[int]]:
        """
        Create non-overlapping splits for each client
        
        Returns:
            Dict mapping client_id to list of data indices
        """
        total_samples = len(self.train_dataset)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        # Split indices among clients
        client_indices = {}
        samples_per_client = total_samples // self.num_clients
        
        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            if client_id == self.num_clients - 1:
                # Last client gets remaining samples
                end_idx = total_samples
            else:
                end_idx = start_idx + samples_per_client
            
            client_indices[client_id] = indices[start_idx:end_idx].tolist()
            print(f"Client {client_id}: {len(client_indices[client_id])} samples")
        
        return client_indices
    
    def get_client_loader(self, client_id: int, shuffle: bool = True) -> DataLoader:
        """
        Get DataLoader for a specific client
        
        Args:
            client_id: Client ID (0 to num_clients-1)
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader for the client
        """
        if client_id not in self.client_indices:
            raise ValueError(f"Client {client_id} not found")
        
        # Create subset for this client
        client_dataset = Subset(self.train_dataset, self.client_indices[client_id])
        
        # Create DataLoader
        loader = DataLoader(
            client_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for MPS compatibility
            pin_memory=False
        )
        
        return loader
    
    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    def get_num_classes(self) -> int:
        """Get number of classes in PathMNIST"""
        return 9  # PathMNIST has 9 tissue types
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return [
            'Adipose', 'Background', 'Debris', 'Lymphoid',
            'Mucosa', 'Smooth Muscle', 'Glandular', 'Stroma', 'Other'
        ]
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get input image shape (C, H, W)"""
        return (3, self.size, self.size)


class PathMNISTWrapper(Dataset):
    """
    Wrapper for PathMNIST to ensure consistent output format
    """
    
    def __init__(self, medmnist_dataset):
        self.dataset = medmnist_dataset
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Convert to tensor if needed
        if not isinstance(img, torch.Tensor):
            # img is likely a PIL Image or numpy array
            if hasattr(img, 'mode'):  # PIL Image
                img = self.transform(img)
            else:  # numpy array
                img = torch.from_numpy(img).float()
                if img.ndim == 2:  # Grayscale
                    img = img.unsqueeze(0).repeat(3, 1, 1)
                elif img.shape[-1] == 3:  # HWC to CHW
                    img = img.permute(2, 0, 1)
                # Normalize
                img = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(img)
        
        # Ensure label is long tensor
        if isinstance(label, np.ndarray):
            label = torch.tensor(label).long().squeeze()
        elif not isinstance(label, torch.Tensor):
            label = torch.tensor(label).long()
        else:
            label = label.long().squeeze()
        
        return img, label


def create_iid_split(
    dataset,
    num_clients: int,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Create IID (independent and identically distributed) data splits
    
    Args:
        dataset: Dataset to split
        num_clients: Number of clients
        seed: Random seed
        
    Returns:
        Dict mapping client_id to indices
    """
    np.random.seed(seed)
    
    total_samples = len(dataset)
    indices = np.random.permutation(total_samples)
    
    client_indices = {}
    samples_per_client = total_samples // num_clients
    
    for client_id in range(num_clients):
        start = client_id * samples_per_client
        end = start + samples_per_client if client_id < num_clients - 1 else total_samples
        client_indices[client_id] = indices[start:end].tolist()
    
    return client_indices


def create_non_iid_split(
    dataset,
    num_clients: int,
    num_shards: int = 200,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Create non-IID data splits (label skew)
    
    Args:
        dataset: Dataset to split
        num_clients: Number of clients
        num_shards: Number of shards to create
        seed: Random seed
        
    Returns:
        Dict mapping client_id to indices
    """
    np.random.seed(seed)
    
    # Get labels
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Sort by label
    sorted_indices = np.argsort(labels)
    
    # Divide into shards
    shard_size = len(dataset) // num_shards
    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else len(dataset)
        shards.append(sorted_indices[start:end].tolist())
    
    # Randomly assign shards to clients
    np.random.shuffle(shards)
    shards_per_client = num_shards // num_clients
    
    client_indices = {}
    for client_id in range(num_clients):
        start = client_id * shards_per_client
        end = start + shards_per_client
        client_shards = shards[start:end]
        client_indices[client_id] = [idx for shard in client_shards for idx in shard]
    
    return client_indices


if __name__ == "__main__":
    # Example usage
    print("Loading PathMNIST for federated learning...")
    
    fed_loader = PathMNISTFederatedLoader(
        num_clients=5,
        batch_size=32,
        download=True,
        seed=42
    )
    
    print(f"\nNumber of classes: {fed_loader.get_num_classes()}")
    print(f"Class names: {fed_loader.get_class_names()}")
    print(f"Input shape: {fed_loader.get_input_shape()}")
    
    # Test loading a batch from client 0
    client_loader = fed_loader.get_client_loader(0)
    images, labels = next(iter(client_loader))
    print(f"\nBatch from client 0:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels in batch: {torch.unique(labels).tolist()}")
    
    # Test loading test data
    test_loader = fed_loader.get_test_loader()
    test_images, test_labels = next(iter(test_loader))
    print(f"\nTest batch:")
    print(f"  Images shape: {test_images.shape}")
    print(f"  Labels shape: {test_labels.shape}")
