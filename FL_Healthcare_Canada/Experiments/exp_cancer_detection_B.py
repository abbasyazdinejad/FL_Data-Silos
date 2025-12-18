"""
Cancer Detection 

"""

import sys
import os

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')


class CancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(np.array(X, dtype=np.float32))
        self.y = torch.LongTensor(np.array(y, dtype=np.int64))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=2):
        super(TabularNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def load_provincial_data(data_dir="data"):
    print("\n" + "-"*70)
    print("LOADING DATASETS")
    print("-"*70)
    
    data_dir = Path(data_dir)
    datasets = {}
    province_names = ['ontario', 'alberta', 'quebec']
    
    print("Looking for:")
    for province in province_names:
        print(f"  - {province}_cancer_data.csv")
    
    for province in province_names:
        filepath = data_dir / f"{province}_cancer_data.csv"
        
        if not filepath.exists():
            print(f"\n✗ Missing {province}_cancer_data.csv")
            print("\nPlease run first:")
            print("  python regenerate_balanced_data.py")
            raise FileNotFoundError(f"Missing {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Fix column names
        if 'Cancer' in df.columns:
            df = df.rename(columns={'Cancer': 'diagnosis'})
        if 'Province' in df.columns:
            df = df.drop('Province', axis=1)
        
        # Ensure numeric
        for col in df.columns:
            if col != 'diagnosis':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN
        df = df.fillna(df.mean(numeric_only=True))
        df = df.dropna(subset=['diagnosis'])
        df['diagnosis'] = df['diagnosis'].astype(int)
        
        # Report
        class_counts = df['diagnosis'].value_counts().sort_index()
        print(f"✓ {province}_cancer_data.csv ({len(df)} samples)")
        print(f"  Class 0: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Class 1: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
        
        datasets[province] = df
    
    print(f"\n✓ Total: {sum(len(df) for df in datasets.values())} samples")
    return datasets


def prepare_data(df, test_size=0.2, random_state=42):
    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def get_class_weights(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)


def train_client(model, train_loader, device, epochs=5, lr=0.001, class_weights=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    num_batches = 0
    
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    if len(np.unique(all_labels)) < 2 or len(np.unique(all_preds)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)
    
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return accuracy, auc, f1, all_preds, all_probs


def federated_averaging(models, dataset_sizes):
    total_samples = sum(dataset_sizes)
    weights = [size / total_samples for size in dataset_sizes]
    
    global_dict = models[0].state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = sum(
            models[i].state_dict()[key].float() * weights[i]
            for i in range(len(models))
        )
    
    return global_dict


def train_siloed_baseline(datasets, test_data, device, num_runs=5):
    print("\n" + "="*70)
    print("SILOED BASELINE (No Federation)")
    print("="*70)
    
    all_aucs = []
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        run_aucs = []
        
        for province, df in datasets.items():
            X_train, X_test, y_train, y_test = prepare_data(df, random_state=42+run)
            class_weights = get_class_weights(y_train)
            
            train_dataset = CancerDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            model = TabularNN(X_train.shape[1]).to(device)
            
            for _ in range(10):
                train_client(model, train_loader, device, epochs=1, class_weights=class_weights)
            
            X_test_all, y_test_all = test_data
            test_dataset = CancerDataset(X_test_all, y_test_all)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            _, auc, _, _, _ = evaluate_model(model, test_loader, device)
            run_aucs.append(auc)
        
        avg_auc = np.mean(run_aucs)
        all_aucs.append(avg_auc)
        print(f"  Average AUC: {avg_auc:.4f}")
    
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs, ddof=1)
    
    print(f"\nSiloed Baseline: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    return all_aucs, mean_auc, std_auc


def run_federated_experiment(datasets, test_data, device, num_rounds=50, 
                            local_epochs=5, num_runs=5):
    print("\n" + "="*70)
    print("FEDERATED LEARNING")
    print("="*70)
    
    all_aucs = []
    all_f1s = []
    
    X_test_all, y_test_all = test_data
    test_dataset = CancerDataset(X_test_all, y_test_all)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    for run in range(num_runs):
        print(f"\n{'='*70}")
        print(f"REPETITION {run+1}/{num_runs}")
        print(f"{'='*70}")
        
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        client_loaders = []
        dataset_sizes = []
        client_class_weights = []
        
        for province, df in datasets.items():
            X_train, _, y_train, _ = prepare_data(df, random_state=42+run)
            class_weights = get_class_weights(y_train)
            client_class_weights.append(class_weights)
            
            train_dataset = CancerDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            client_loaders.append(train_loader)
            dataset_sizes.append(len(train_dataset))
        
        global_model = TabularNN(X_train.shape[1]).to(device)
        
        print(f"\n{'─'*70}")
        print("Training Progress")
        print(f"{'─'*70}")
        
        start_time = time.time()
        
        for round_num in tqdm(range(1, num_rounds + 1), desc="FL Rounds"):
            client_models = []
            round_losses = []
            
            for train_loader, class_weights in zip(client_loaders, client_class_weights):
                client_model = TabularNN(X_train.shape[1]).to(device)
                client_model.load_state_dict(global_model.state_dict())
                
                loss = train_client(client_model, train_loader, device, 
                                  epochs=local_epochs, lr=0.001, class_weights=class_weights)
                
                client_models.append(client_model)
                round_losses.append(loss)
            
            global_dict = federated_averaging(client_models, dataset_sizes)
            global_model.load_state_dict(global_dict)
            
            if round_num % 10 == 0 or round_num == num_rounds:
                accuracy, auc, f1, _, _ = evaluate_model(global_model, test_loader, device)
                avg_loss = np.mean(round_losses)
                tqdm.write(f"  Round {round_num:3d}/{num_rounds}: "
                          f"Loss={avg_loss:.4f}, AUC={auc:.4f}, Acc={accuracy:.4f}")
        
        training_time = time.time() - start_time
        
        final_acc, final_auc, final_f1, _, _ = evaluate_model(global_model, test_loader, device)
        
        all_aucs.append(final_auc)
        all_f1s.append(final_f1)
        
        print(f"\nRun {run+1}: AUC={final_auc:.4f}, Acc={final_acc:.4f}, F1={final_f1:.4f}")
        print(f"Time: {training_time/60:.1f} minutes")
    
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs, ddof=1)
    mean_f1 = np.mean(all_f1s)
    
    print(f"\n{'='*70}")
    print(f"FEDERATED RESULTS")
    print(f"{'='*70}")
    print(f"AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"F1:  {mean_f1:.4f}")
    
    return all_aucs, mean_auc, std_auc


def run():
    print("\n" + "="*70)
    print("CANCER DETECTION - Use Case 1")
    print("="*70)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Metal (MPS) GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using GPU")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        datasets = load_provincial_data("data")
    except FileNotFoundError as e:
        print(f"\n{str(e)}")
        return None
    
    # Combined test set
    all_test_X = []
    all_test_y = []
    
    for df in datasets.values():
        _, X_test, _, y_test = prepare_data(df)
        all_test_X.append(X_test)
        all_test_y.append(y_test)
    
    X_test_combined = np.vstack(all_test_X)
    y_test_combined = np.concatenate(all_test_y)
    test_data = (X_test_combined, y_test_combined)
    
    print(f"\nCombined test set: {len(y_test_combined)} samples")
    
    # Run experiments
    siloed_aucs, siloed_mean, siloed_std = train_siloed_baseline(
        datasets, test_data, device, num_runs=5
    )
    
    fl_aucs, fl_mean, fl_std = run_federated_experiment(
        datasets, test_data, device, num_rounds=50, local_epochs=5, num_runs=5
    )
    
    # Statistical analysis
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}")
    
    t_stat, p_value = stats.ttest_rel(fl_aucs, siloed_aucs)
    improvement = ((fl_mean - siloed_mean) / siloed_mean) * 100
    
    print(f"\nFederated Learning: {fl_mean:.4f} ± {fl_std:.4f}")
    print(f"Siloed Baseline:    {siloed_mean:.4f} ± {siloed_std:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"p-value: {p_value:.4f} {'✓ Significant' if p_value < 0.05 else '✗ Not significant'}")
    
    # Save results
    results = {
        'Use Case': 'Cancer Detection',
        'FL_AUC_mean': fl_mean,
        'FL_AUC_std': fl_std,
        'Siloed_AUC_mean': siloed_mean,
        'Siloed_AUC_std': siloed_std,
        'Improvement_%': improvement,
        'p_value': p_value,
        'Significant': p_value < 0.05,
        'Num_provinces': len(datasets),
        'Total_samples': sum(len(df) for df in datasets.values()),
        'Num_runs': 5
    }
    
    df_results = pd.DataFrame([results])
    output_path = results_dir / "cancer_detection_results_real.csv"
    df_results.to_csv(output_path, index=False)
    
    print(f"\n✓ Results saved to {output_path}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    results = run()
