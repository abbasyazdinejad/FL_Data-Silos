"""
Demo Data generation and loading modules 
"""

from .synthetic_generator import (
    ProvincialDataGenerator,
    generate_provincial_data,
    generate_cancer_datasets,
    generate_pandemic_datasets,
    generate_rare_disease_datasets,
    save_datasets_as_csv
)

from .data_loader import (
    TabularDataset,
    FederatedDataLoader,
    load_from_csv_files
)

from .pathmnist_loader import (
    load_pathmnist,
    PathMNISTFederatedLoader,
    PathMNISTWrapper,
    create_iid_split,
    create_non_iid_split
)

__all__ = [
    # Synthetic data generation
    'ProvincialDataGenerator',
    'generate_provincial_data',
    'generate_cancer_datasets',
    'generate_pandemic_datasets',
    'generate_rare_disease_datasets',
    'save_datasets_as_csv',
    
    # Federated data loading
    'TabularDataset',
    'FederatedDataLoader',
    'load_from_csv_files',
    
    # PathMNIST
    'load_pathmnist',
    'PathMNISTFederatedLoader',
    'PathMNISTWrapper',
    'create_iid_split',
    'create_non_iid_split'
]

__version__ = '1.0.0'
__author__ = 'Abbas Yazdinejad and Jude Kong'
