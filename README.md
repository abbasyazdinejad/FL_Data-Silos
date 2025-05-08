# Federated Learning for Cancer Detection

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

## 🔐 Privacy

We simulate differential privacy using DP-SGD (ε ≈ 1.0) and non-IID partitioning to reflect real-world deployments.

## 📜 License

MIT License
