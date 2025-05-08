# How Federated Learning Can Unlock Canada’s Public Health Potential

This repository contains synthetic and benchmark datasets used in our research on privacy-preserving federated learning for cancer classification, as presented in our [RSIF submission].

## 📂 Contents

- `notebooks/` – Jupyter notebooks for:
  - Generating synthetic cancer data across provinces
  - Federated training on PathMNIST (binary and multiclass)
- `data/` – CSV datasets for three provinces + full combined dataset

## 🚀 How to Use

1. Clone this repo:
git clone https://github.com/your-username/FederatedCancerFL.git

2. Open notebooks:
- `Synthetic Data.ipynb`: Generate and explore synthetic provincial data.
- `PathMNIST_federated_learning2.ipynb`: Train and evaluate a federated model.

3. Python scripts:
 - DP.py: Simulate privacy-preserving federated learning results and generate comparative performance metrics across privacy levels (ε).


## 🔐 Privacy

We simulate differential privacy using DP-SGD (ε ≈ 1.0) and non-IID partitioning to reflect real-world deployments.

# 📊 Dataset Description – Synthetic Cancer Detection

This folder contains a synthetic dataset designed to support federated learning research focused on cancer detection across Canadian provinces. The data simulates **non-IID distributions** by applying province-specific bias to the cancer probability function.

## 📁 Files

- `ontario_cancer_data.csv`
- `alberta_cancer_data.csv`
- `quebec_cancer_data.csv`
- `synthetic_cancer_dataset.csv` – Combined dataset (30,000 records across all provinces)

## 📋 Schema

| Column       | Description                                                                  |
|--------------|------------------------------------------------------------------------------|
| `Province`   | Province name (`Ontario`, `Alberta`, or `Quebec`)                            |
| `Age`        | Patient age (mean = 60, SD = 10)                                              |
| `TumorSize`  | Tumor size in centimeters (mean = 3.0, SD = 1.0)                              |
| `Biomarker`  | Simulated biomarker level (mean = 100, SD = 25)                               |
| `Cancer`     | Binary label: `1` = cancer, `0` = no cancer                                   |

## 🔄 Bias Simulation by Province

To emulate real-world heterogeneity in patient risk profiles, the cancer generation probability was adjusted with the following bias factors:

- **Ontario**: Balanced distribution (bias = 0)
- **Alberta**: Lower prevalence (bias = -10)
- **Quebec**: Higher prevalence (bias = +10)

These biases are introduced using a logistic probability function during synthetic data generation and enable evaluation of federated learning performance under realistic inter-client skew.

## 🔐 Use Case

This dataset supports experiments comparing:
- Centralized AI vs. Federated Learning (FL)
- Federated Learning with Privacy (FL + DP)
- Performance and fairness tradeoffs under heterogeneous data conditions

## 🧪 Additional Dataset: PathMNIST (Benchmark)

We also use the publicly available [PathMNIST](https://medmnist.com/) dataset from the MedMNIST v2 collection to benchmark federated learning under realistic histopathology data settings.

PathMNIST includes 107,180 RGB images (28×28) of colorectal cancer tissue from 9 classes:

- **0:** Adipose
- **1:** Background
- **2:** Debris
- **3:** Lymphocytes
- **4:** Mucus
- **5:** Smooth Muscle
- **6:** Normal Colon Mucosa
- **7:** Cancer-Associated Stroma
- **8:** Colorectal Adenocarcinoma Epithelium

### Usage in This Project

- Federated simulation is performed with **3 non-IID clients**.
- Models include **ResNet-18** with and without **differential privacy**.
- We evaluate:
  - **Multi-class classification** (9 classes)
  - **Binary reduction** (class 0 vs. class 1) for fast convergence
- Performance metrics:
  - **Accuracy**
  - **Macro-F1 Score**
  - **Differential privacy budget (ε ≈ 1.0)**

The PathMNIST dataset is downloaded automatically via the `medmnist` Python package.


## 📜 License

MIT License
