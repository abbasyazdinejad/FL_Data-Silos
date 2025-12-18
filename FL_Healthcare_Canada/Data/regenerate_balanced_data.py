"""
Regenerate Balanced Cancer Data - 50/50 Split

This creates properly balanced cancer datasets for training.
Run this once to fix the extreme imbalance issue.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_balanced_cancer_data(n_samples, province_name, cancer_ratio=0.5, seed=42):
    """
    Generate balanced synthetic cancer data
    
    Args:
        n_samples: Total number of samples
        province_name: Name of province
        cancer_ratio: Ratio of cancer cases (default 0.5 for 50/50 balance)
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Calculate samples per class
    n_cancer = int(n_samples * cancer_ratio)
    n_no_cancer = n_samples - n_cancer
    
    print(f"\nGenerating {province_name}:")
    print(f"  Total: {n_samples} samples")
    print(f"  Cancer (Class 1): {n_cancer} ({cancer_ratio*100:.1f}%)")
    print(f"  No Cancer (Class 0): {n_no_cancer} ({(1-cancer_ratio)*100:.1f}%)")
    
    # Generate features for cancer cases (Class 1)
    # Higher age, larger tumors, higher biomarkers
    age_cancer = np.random.normal(65, 10, n_cancer).clip(30, 90)
    tumor_cancer = np.random.normal(3.5, 1.0, n_cancer).clip(1, 5)
    biomarker_cancer = np.random.normal(110, 20, n_cancer).clip(50, 150)
    diagnosis_cancer = np.ones(n_cancer, dtype=int)
    
    # Generate features for no cancer cases (Class 0)
    # Lower age, smaller tumors, lower biomarkers
    age_no_cancer = np.random.normal(55, 12, n_no_cancer).clip(25, 85)
    tumor_no_cancer = np.random.normal(2.0, 0.8, n_no_cancer).clip(0.5, 4)
    biomarker_no_cancer = np.random.normal(75, 15, n_no_cancer).clip(40, 120)
    diagnosis_no_cancer = np.zeros(n_no_cancer, dtype=int)
    
    # Combine
    age = np.concatenate([age_cancer, age_no_cancer])
    tumor_size = np.concatenate([tumor_cancer, tumor_no_cancer])
    biomarker = np.concatenate([biomarker_cancer, biomarker_no_cancer])
    diagnosis = np.concatenate([diagnosis_cancer, diagnosis_no_cancer])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    age = age[indices]
    tumor_size = tumor_size[indices]
    biomarker = biomarker[indices]
    diagnosis = diagnosis[indices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Province': province_name,
        'Age': age,
        'TumorSize': tumor_size,
        'Biomarker': biomarker,
        'Cancer': diagnosis
    })
    
    return df


def main():
    """Generate balanced cancer datasets for all provinces"""
    print("="*70)
    print("GENERATING BALANCED CANCER DATA (50/50 SPLIT)")
    print("="*70)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate datasets for each province
    provinces = {
        'ontario': 20000,
        'alberta': 10000,
        'quebec': 15000
    }
    
    for province, n_samples in provinces.items():
        df = generate_balanced_cancer_data(
            n_samples=n_samples,
            province_name=province.capitalize(),
            cancer_ratio=0.5,  # 50/50 balance
            seed=42
        )
        
        # Save to CSV
        output_path = data_dir / f"{province}_cancer_data.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved to {output_path}")
        
        # Verify
        class_counts = df['Cancer'].value_counts()
        print(f"  ✓ Verified: Class 0={class_counts[0]}, Class 1={class_counts[1]}")
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print("="*70)
    print(f"\n✓ Generated {sum(provinces.values())} total samples")
    print("✓ All datasets are 50/50 balanced")
    print("✓ Ready for training!")
    print("\nNext steps:")
    print("1. Run: python Experiments/run_all_experiments_WORKING.py")
    print("2. Watch AUC improve to 0.9+ during training!")


if __name__ == "__main__":
    main()
