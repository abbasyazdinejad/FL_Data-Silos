# Demo FL Healthcare Canada: Federated Learning with AI Governance

**Breaking Interprovincial Data Silos: How Federated Learning Can Unlock Canada's Public Health Potential**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Demo of implementation of the federated learning + AI governance framework for Canadian healthcare,
---

##  Overview

This repository provides a **complete, production-ready implementation** of:

1. **Federated Learning (FL)** - FedAvg, FedProx algorithms
2. **AI Governance** - Consent, audit, fairness, compliance
3. **Differential Privacy** - (ε, δ)-DP with DP-SGD
4. **Evaluation Framework** - Metrics, statistical tests, visualizations
5. **Real-world Use Cases** - Cancer detection, medical imaging

### Key Results (Table 5 from Paper)

| Use Case | Performance Gain | Privacy | Governance |
|----------|-----------------|---------|------------|
| **Cancer Detection** | +4.2% AUC (0.92±0.01 vs 0.88±0.02, p<0.05) | 100% data localization | 2.3% bias flags |
| **Medical Imaging** | 91.3±0.4% accuracy, F1=91.14 | ε=1.0 maintained | 0 violations |

---

##  Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU)
- Apple Metal (optional, for M1/M2/M3/M4)

### Quick Install

```bash
# Clone repository
git clone https://github.com/your-username/FL_Healthcare_Canada.git
cd FL_Healthcare_Canada

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
medmnist>=2.2.0
python-docx>=0.8.11
openpyxl>=3.1.0
pyyaml>=6.0
tqdm>=4.66.0
```

---

##  Quick Start

### 1. Generate Data

```bash
# Generate synthetic provincial cancer datasets
python data/synthetic_generator.py
```

This creates:
- `ontario_cancer_data.csv` (20,000 samples)
- `alberta_cancer_data.csv` (10,000 samples)
- `quebec_cancer_data.csv` (15,000 samples)

### 2. Run Cancer Detection Experiment

```python
from experiments import run_cancer_detection

# Run experiment with 5 repetitions
results = run_cancer_detection()
```

Output:
```
============================================================
CANCER DETECTION EXPERIMENT - Use Case 1
Ontario, Alberta, Quebec
============================================================

Federated Learning AUC: 0.9180 ± 0.0084
Siloed Baseline AUC: 0.8800 ± 0.0084
Improvement: +4.3%

Paired t-test: t=30.48, p=0.0001
Significant (p<0.05): Yes ✓

✓ Results saved to results/tables/cancer_detection_results.csv
```

### 3. Run PathMNIST Experiment

```python
from experiments import run_pathmnist

# Run medical imaging experiment
results = run_pathmnist()
```

### 4. Run All Experiments (Generate Table 5)

```python
from experiments import run_all_experiments

# Generate complete Table 5
table5 = run_all_experiments()
```

---

## 📚 Detailed Usage

### Federated Learning

#### FedAvg Training

```python
from federated import FedAvgTrainer, FederatedClient
from data_loader import FederatedDataLoader
import torch

# Setup device
device = torch.device('mps')  # or 'cuda', 'cpu'

# Load data
data_loader = FederatedDataLoader(...)

# Create clients
clients = [
    FederatedClient('ontario', model, ontario_loader, device),
    FederatedClient('alberta', model, alberta_loader, device),
    FederatedClient('quebec', model, quebec_loader, device)
]

# Create trainer
trainer = FedAvgTrainer(model, clients, device, test_loader=test_loader)

# Train
history = trainer.train(num_rounds=50, evaluate_every=1, verbose=True)

# Get trained model
trained_model = trainer.get_global_model()
```

#### FedProx Training (for non-IID data)

```python
from federated import FedProxTrainer, FedProxClient

# Create FedProx clients
clients = [
    FedProxClient('ontario', model, ontario_loader, device, mu=0.01),
    FedProxClient('alberta', model, alberta_loader, device, mu=0.01),
    FedProxClient('quebec', model, quebec_loader, device, mu=0.01)
]

# Train with proximal term
trainer = FedProxTrainer(model, clients, device, mu=0.01)
history = trainer.train(num_rounds=50)
```

### Differential Privacy

#### DP-SGD Training

```python
from privacy import PrivacyEngine
import torch.optim as optim

# Create model and optimizer
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create privacy engine
privacy_engine = PrivacyEngine(
    model=model,
    optimizer=optimizer,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    delta=1e-5,
    target_epsilon=2.0
)

# Training loop
for data, target in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    
    # Apply DP before optimizer step
    privacy_engine.step(batch_size=len(data))
    optimizer.step()

# Check privacy spent
privacy_spent = privacy_engine.get_privacy_spent(
    dataset_size=10000,
    batch_size=32
)
print(f"Privacy spent: ε={privacy_spent['epsilon']:.2f}, δ={privacy_spent['delta']:.2e}")
```

### AI Governance

#### Complete Governance Pipeline

```python
from governance import create_governance_pipeline

# Create all governance components
governance = create_governance_pipeline()

# Use in FL training
for round_num in range(1, 51):
    # 1. Check consent
    if governance['consent_manager'].check_consent('ontario', 'training'):
        
        # 2. Train
        result = client.train_local(global_params, round_num)
        
        # 3. Log audit event
        governance['audit_logger'].log_training_event(
            'ontario', round_num, result['loss'], result['num_samples']
        )
        
        # 4. Check fairness (after evaluation)
        fairness = governance['fairness_auditor'].audit_predictions(
            y_true, y_pred
        )
        
        # 5. Check compliance
        compliance = governance['compliance_checker'].check_privacy_compliance(
            epsilon_spent=1.0
        )

# Verify audit chain
print(f"Audit chain valid: {governance['audit_logger'].verify_chain()}")

# Get governance overhead
from governance import get_governance_overhead
overhead = get_governance_overhead(num_clients=3)
print(f"Governance overhead: {overhead['overhead_percentage']:.1f}%")
```

### Evaluation

#### Computing Metrics

```python
from evaluation import compute_metrics, paired_t_test, plot_confusion_matrix

# Compute comprehensive metrics
metrics = compute_metrics(y_true, y_pred, y_prob, average='binary')
print(f"AUC: {metrics['auc']:.4f}")
print(f"F1: {metrics['f1']:.4f}")

# Statistical significance
test_result = paired_t_test(fl_results, siloed_results, alpha=0.05)
print(f"p-value: {test_result['p_value']:.4f}")
print(f"Significant: {test_result['significant']}")

# Visualization
plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png')
```

---

## 🔬 Reproducing Paper Results

### Step 1: Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Set random seed for reproducibility
python -c "from utilities import make_reproducible; make_reproducible(seed=42)"
```

### Step 2: Generate Data

```bash
# Generate synthetic provincial data
python data/synthetic_generator.py

# This creates:
# - ontario_cancer_data.csv
# - alberta_cancer_data.csv
# - quebec_cancer_data.csv
```

### Step 3: Run Experiments

```bash
# Run all experiments and generate Table 5
python experiments/run_all_experiments.py
```

### Expected Output

```
==================================================================
TABLE 5: MAIN RESULTS SUMMARY
==================================================================

Use Case                     Performance Gain              Privacy Protection
Cancer Detection             +4.3% AUC (0.92±0.01 vs      100% data localization
                             0.88±0.01)
Medical Imaging (PathMNIST)  91.3±0.2% accuracy, 91.1 F1  ε=1.0 maintained

✓ Saved CSV to: results/tables/table5_main_results.csv
✓ Saved text to: results/tables/table5_main_results.txt
```

### Verifying Results

```python
import pandas as pd

# Load results
results = pd.read_csv('results/tables/table5_main_results.csv')

# Compare with paper
# Cancer Detection: +4.2% AUC (paper) vs +4.3% (our results)
# PathMNIST: 91.3±0.4% (paper) vs 91.3±0.2% (our results)
```

---

## 📂 Project Structure

```
FL_Healthcare_Canada/
├── data/                       # Data loading and generation
│   ├── __init__.py
│   ├── synthetic_generator.py  # Generate provincial data
│   ├── data_loader.py         # Load tabular data
│   └── pathmnist_loader.py    # Load PathMNIST
├── models/                     # Neural network models
│   ├── __init__.py
│   ├── cnn_models.py          # SimpleCNN, ResNet18
│   └── federated_models.py    # TabularNN, FederatedModel
├── federated/                  # Federated learning core
│   ├── __init__.py
│   ├── aggregation.py         # FedAvg, median, secure
│   ├── client.py              # FederatedClient, FedProxClient
│   ├── server.py              # FederatedServer
│   ├── fed_avg.py             # FedAvg trainer
│   └── fed_prox.py            # FedProx trainer
├── privacy/                    # Differential privacy
│   ├── __init__.py
│   ├── differential_privacy.py # DP mechanisms
│   └── privacy_engine.py      # DP-SGD engine
├── governance/                 # AI governance
│   ├── __init__.py
│   ├── consent_manager.py     # SMART-on-FHIR consent
│   ├── audit_logger.py        # Hash-chained logging
│   ├── fairness_auditor.py    # Bias detection
│   └── compliance_checker.py  # PIPEDA/PHIPA/OCAP®
├── evaluation/                 # Evaluation framework
│   ├── __init__.py
│   ├── metrics.py             # Accuracy, AUC, F1
│   ├── statistical_tests.py   # t-tests, CIs
│   └── visualization.py       # Plots and figures
├── experiments/                # Experiment runners
│   ├── __init__.py
│   ├── exp_cancer_detection.py
│   ├── exp_pathmnist.py
│   └── run_all_experiments.py
├── utilities/                  # Helper utilities
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── logger.py              # Logging utilities
│   └── reproducibility.py     # Reproducibility tools
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---


## 🔧 Configuration

### Creating a Custom Configuration

```python
from utilities import ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    num_rounds=100,
    local_epochs=10,
    batch_size=64,
    learning_rate=0.001,
    epsilon=1.5,
    delta=1e-5
)

# Save configuration
config_mgr = ConfigManager()
config_mgr.config = config
config_mgr.save("my_config.json")

# Load configuration
config_mgr.load("my_config.json")
```

### Using Predefined Configurations

```python
from utilities import get_config

# Get recommended config for use case
config = get_config('cancer_detection')
print(f"Rounds: {config.num_rounds}")
print(f"Epsilon: {config.epsilon}")
```

---

## Reproducing Specific Results

### Results

#### Cancer Detection
```bash
python experiments/exp_cancer_detection.py
```

Expected: AUC=0.92±0.01 (FL) vs 0.88±0.02 (Siloed), p<0.05

#### PathMNIST Benchmark
```bash
python experiments/exp_pathmnist.py
```

Expected: Accuracy=91.3±0.4%, F1=91.14, ε=1.0

### Statistical Analysis

```python
from evaluation import paired_t_test, compute_summary_statistics

# Run 5 repetitions
fl_aucs = [0.92, 0.91, 0.93, 0.92, 0.91]
siloed_aucs = [0.88, 0.89, 0.87, 0.88, 0.89]

# Statistical test
result = paired_t_test(fl_aucs, siloed_aucs)
print(f"t={result['t_statistic']:.2f}, p={result['p_value']:.4f}")

# Summary statistics
fl_stats = compute_summary_statistics(fl_aucs)
print(f"FL: {fl_stats['mean']:.3f} ± {fl_stats['std']:.3f}")
print(f"95% CI: [{fl_stats['ci_95_lower']:.3f}, {fl_stats['ci_95_upper']:.3f}]")
```

---

## Testing

### Unit Tests

```bash
# Test individual modules
python data/data_loader.py
python federated/aggregation.py
python privacy/differential_privacy.py
python governance/consent_manager.py
python evaluation/metrics.py
```

### Integration Tests

```bash
# Test complete pipeline
python experiments/run_all_experiments.py
```

### Reproducibility Verification

```python
from utilities import verify_reproducibility

# Run experiment twice with same seed
results1 = run_experiment(seed=42)
results2 = run_experiment(seed=42)

# Verify results match
verify_reproducibility(results1, results2, tolerance=1e-6)
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce batch size
config.batch_size = 16  # instead of 32
```

#### 2. MPS Backend Issues (Apple Silicon)
```python
# Set environment variable
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

#### 3. medmnist Installation
```bash
pip install medmnist --upgrade
```

#### 4. Import Errors
```python
# Add project root to Python path
import sys
sys.path.insert(0, '/path/to/FL_Healthcare_Canada')
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Results Interpretation and Reproducibility Statement.


All quantitative results reported in this study are derived from controlled simulation experiments designed to evaluate the methodological feasibility and governance instrumentation of federated learning in a decentralized healthcare context. Except for the PathMNIST benchmark, all datasets are synthetically generated under explicitly stated assumptions regarding class prevalence, heterogeneity, and noise. Reported performance metrics represent mean values aggregated over multiple independent runs with randomized initialization.

Due to stochastic training dynamics, synthetic data regeneration, thresholding effects, and environment-specific dependencies, individual executions of the accompanying reference code may yield performance metrics that differ from the reported averages. Such variation is expected and does not affect the comparative conclusions of the study, which focus on relative performance trends, governance impacts, and statistical significance rather than exact numerical replication.

The provided code is intended as a transparent reference implementation of the proposed framework and experimental design, not as a deterministic reproduction pipeline for all reported numerical results.

---


