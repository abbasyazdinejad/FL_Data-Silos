"""
Run All Experiments - BALANCED DATA VERSION

Uses properly balanced 50/50 cancer data.
Clean training that actually works!
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from pathlib import Path


def print_header():
    print("\n" + "="*80)
    print("FEDERATED LEARNING + AI GOVERNANCE")
    print("Table 5 Results - BALANCED DATA (50/50 Split)")
    print("="*80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_system_info():
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("Device: Apple Silicon GPU")
    elif torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")


def run_cancer_detection():
    print("\n" + "="*80)
    print("EXPERIMENT 1/2: CANCER DETECTION")
    print("="*80)
    
    try:
        from exp_cancer_detection_BALANCED import run
        return run()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_pathmnist():
    print("\n" + "="*80)
    print("EXPERIMENT 2/2: PATHMNIST")
    print("="*80)
    
    try:
        from exp_pathmnist_REAL_FIXED import run
        return run()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_table5(cancer_results, pathmnist_results):
    print("\n" + "="*80)
    print("TABLE 5: MAIN RESULTS")
    print("="*80)
    
    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    table_data = []
    
    if cancer_results:
        table_data.append({
            'Use Case': 'Cancer Detection',
            'Performance': f"+{cancer_results['Improvement_%']:.1f}% AUC "
                          f"({cancer_results['FL_AUC_mean']:.2f}±{cancer_results['FL_AUC_std']:.2f})",
            'p-value': f"{cancer_results['p_value']:.4f}"
        })
    
    if pathmnist_results:
        table_data.append({
            'Use Case': 'PathMNIST',
            'Performance': f"{pathmnist_results['Accuracy_mean']*100:.1f}±{pathmnist_results['Accuracy_std']*100:.1f}%",
            'p-value': 'N/A'
        })
    
    df_table5 = pd.DataFrame(table_data)
    print("\n" + df_table5.to_string(index=False))
    
    csv_path = results_dir / "table5_final.csv"
    df_table5.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")
    
    return df_table5


def main():
    start_time = time.time()
    
    print_header()
    print_system_info()
    
    cancer_results = run_cancer_detection()
    pathmnist_results = run_pathmnist()
    
    if cancer_results or pathmnist_results:
        generate_table5(cancer_results, pathmnist_results)
    
    elapsed = (time.time() - start_time) / 60
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Time: {elapsed:.1f} minutes")
    print(f"Completed: {int(bool(cancer_results)) + int(bool(pathmnist_results))}/2")
    
    if cancer_results and pathmnist_results:
        print("\n✓ ALL EXPERIMENTS COMPLETED!")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
