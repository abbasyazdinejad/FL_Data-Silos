# How Federated Learning Can Unlock Canada’s Public Health Potential

This repository contains the code, datasets, and experimental workflows supporting **privacy-preserving and governance-aware federated learning** for Canadian healthcare applications.

## 📂 Contents

- `notebooks/` – Jupyter notebooks for:
  - Generating **synthetic non-IID cancer data** across provinces (Ontario, Alberta, Quebec)
  - **Federated training** on PathMNIST for cancer tissue classification
- `data/` – CSV datasets for three provinces + combined dataset
- `DP.py` – Implements differential privacy (DP-SGD) and performance–privacy trade-off analysis
- `equity_bias_simulation.py` – Models participation bias effects (dropout, noisy updates) and fairness impact

## 🚀 How to Use

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-username/FederatedCancerFL.git
   cd FederatedCancerFL
   ```

2. **Run notebooks**
   - `Synthetic Data.ipynb`: Generate and explore synthetic provincial datasets.  
   - `PathMNIST_federated_learning2.ipynb`: Train and evaluate federated models (with/without differential privacy).  

3. **Python scripts**
   - `DP.py`: Simulate privacy-preserving training and generate comparative performance metrics across privacy levels (ε).  
   - `equity_bias_simulation.py`: Reproduce fairness–performance trade-offs described in Section 4.3.

---

## 🔐 Privacy

Federated learning experiments implement **Differentially Private SGD** with privacy budgets of ε ≈ 1.0.  
All data used are synthetic or publicly available benchmarks (MedMNIST/PathMNIST).  
No identifiable or clinical data are included in this repository.

---

## 🧭 Equity Simulation

`equity_bias_simulation.py` reproduces the fairness–performance trade-off discussed in Section 4.3 of the paper.  
It simulates dropout and noisy-update scenarios among provinces to quantify how unequal participation affects model accuracy and fairness.

---

## 📊 Dataset Description – Synthetic Cancer Detection

This synthetic dataset emulates inter-provincial variation in cancer prevalence and patient demographics to study non-IID behavior in federated systems.

### 📁 Files
- `ontario_cancer_data.csv`
- `alberta_cancer_data.csv`
- `quebec_cancer_data.csv`
- `synthetic_cancer_dataset.csv` – Combined dataset (≈ 30,000 records)

### 📋 Schema
| Column | Description |
|--------|-------------|
| `Province` | Province name (`Ontario`, `Alberta`, `Quebec`) |
| `Age` | Patient age (mean = 60 ± 10) |
| `TumorSize` | Tumor size in cm (mean = 3.0 ± 1.0) |
| `Biomarker` | Simulated biomarker level (mean = 100 ± 25) |
| `Cancer` | Binary label: `1` = cancer, `0` = no cancer |

### 🔄 Bias Simulation by Province
- **Ontario**: Balanced distribution (bias = 0)  
- **Alberta**: Lower prevalence (bias = −10)  
- **Quebec**: Higher prevalence (bias = +10)  

These bias factors are applied during data generation using a logistic probability model.

---

## 🧪 Additional Dataset: PathMNIST Benchmark

We use the publicly available [PathMNIST](https://medmnist.com/) dataset from the **MedMNIST v2** collection to benchmark FL under real histopathology images.  
It contains 107,180 RGB (28×28) images from 9 tissue classes of colorectal cancer.  

**Experiments include:**
- 3 non-IID clients (simulating provincial data)
- **ResNet-18** models with and without DP
- Evaluation metrics:
  - Accuracy
  - Macro-F1 Score
  - Privacy budget (ε ≈ 1.0)

---

## ⚖️ Use Cases

This repository supports comparison of:
- Centralized vs. Federated Learning  
- Federated Learning + Differential Privacy  
- Fairness and performance trade-offs under non-IID conditions  
- Equity implications of participation bias across jurisdictions
- This is v1.0-rsos
- This repository provides supporting code and example simulations for the study
“Breaking Interprovincial Data Silos: How Federated Learning Can Unlock Canada’s Public Health Potential.”
The code is intended to illustrate the proposed federated learning framework, governance instrumentation, and experimental design.
Due to stochastic training, synthetic data generation, and environment-specific dependencies, exact numerical reproduction of all reported results is not guaranteed.
---

## 📜 License
MIT License © 2025

---
