"""
Synthetic data generation for provincial datasets

"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split


class ProvincialDataGenerator:
    """
    Generate synthetic provincial healthcare data matching 
    """
    
    # Provincial demographics from 2021 Statistics Canada
    DEMOGRAPHICS = {
        'ontario': {
            'population': 14_800_000,
            'median_age': 41.2,
            'female_pct': 51.1,
            'disease_prevalence': 0.052,  # 5.2% (525 per 100k)
            'samples': 20000
        },
        'alberta': {
            'population': 4_500_000,
            'median_age': 38.4,
            'female_pct': 49.8,
            'disease_prevalence': 0.051,  # 5.1% (508 per 100k)
            'samples': 10000
        },
        'quebec': {
            'population': 8_700_000,
            'median_age': 42.6,
            'female_pct': 50.4,
            'disease_prevalence': 0.054,  # 5.4% (544 per 100k)
            'samples': 15000
        }
    }
    
    def __init__(self, seed: int = 42):
        """
        Initialize data generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def generate_province_data(
        self, 
        province: str,
        num_features: int = 10,
        introduce_heterogeneity: bool = True
    ) -> pd.DataFrame:
        """
        Generate synthetic data for a specific province
        
        Args:
            province: Province name ('ontario', 'alberta', 'quebec')
            num_features: Number of clinical features
            introduce_heterogeneity: Whether to introduce non-IID distributions
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        if province.lower() not in self.DEMOGRAPHICS:
            raise ValueError(f"Unknown province: {province}")
        
        demo = self.DEMOGRAPHICS[province.lower()]
        n_samples = demo['samples']
        
        # Generate age distribution (Gaussian around median age)
        age_mean = demo['median_age']
        age_std = 15.0
        ages = np.random.normal(age_mean, age_std, n_samples)
        ages = np.clip(ages, 18, 95)  # Clip to realistic range
        
        # Generate sex (binary: 0=male, 1=female)
        female_prob = demo['female_pct'] / 100
        sex = np.random.binomial(1, female_prob, n_samples)
        
        # Generate BMI (correlated with age)
        bmi_base = 26.0
        bmi_age_effect = 0.05 * (ages - age_mean)
        bmi_noise = np.random.normal(0, 4.0, n_samples)
        bmi = bmi_base + bmi_age_effect + bmi_noise
        bmi = np.clip(bmi, 15, 45)
        
        # Generate blood pressure (systolic)
        bp_mean = 120 + 0.3 * (ages - age_mean)
        bp_noise = np.random.normal(0, 15, n_samples)
        blood_pressure = bp_mean + bp_noise
        blood_pressure = np.clip(blood_pressure, 90, 180)
        
        # Generate glucose level
        glucose_mean = 100 + 0.15 * (ages - age_mean) + 0.5 * (bmi - bmi_base)
        glucose_noise = np.random.normal(0, 20, n_samples)
        glucose = glucose_mean + glucose_noise
        glucose = np.clip(glucose, 70, 200)
        
        # Generate additional clinical features with provincial variation
        if introduce_heterogeneity:
            # Add provincial-specific perturbations (±10-20%)
            provincial_shift = self._get_provincial_shift(province)
        else:
            provincial_shift = 1.0
        
        features = []
        for i in range(num_features - 5):  # -5 because we have age, sex, bmi, bp, glucose
            feature_mean = 50 * provincial_shift
            feature_std = 15 * provincial_shift
            feature = np.random.normal(feature_mean, feature_std, n_samples)
            features.append(feature)
        
        # Generate diagnosis labels with class imbalance
        # Higher risk based on age, BMI, and other factors
        risk_score = (
            0.3 * (ages - age_mean) / age_std +
            0.3 * (bmi - bmi_base) / 4.0 +
            0.2 * (blood_pressure - 120) / 15.0 +
            0.2 * (glucose - 100) / 20.0
        )
        
        # Apply provincial disease prevalence
        prevalence = demo['disease_prevalence']
        # Use logistic function to convert risk scores to probabilities
        prob_disease = 1 / (1 + np.exp(-risk_score))
        # Scale to match target prevalence
        threshold = np.percentile(prob_disease, (1 - prevalence) * 100)
        diagnosis = (prob_disease > threshold).astype(int)
        
        # Create DataFrame
        data = {
            'age': ages,
            'sex': sex,
            'bmi': bmi,
            'blood_pressure': blood_pressure,
            'glucose': glucose
        }
        
        # Add additional features
        for i, feat in enumerate(features):
            data[f'feature_{i+6}'] = feat
        
        data['diagnosis'] = diagnosis
        data['province'] = province.lower()
        
        df = pd.DataFrame(data)
        
        return df
    
    def _get_provincial_shift(self, province: str) -> float:
        """Get provincial-specific shift factor for heterogeneity"""
        shifts = {
            'ontario': 1.0,      # Baseline
            'alberta': 0.9,      # 10% lower on average
            'quebec': 1.1        # 10% higher on average
        }
        return shifts.get(province.lower(), 1.0)
    
    def compute_divergence_metrics(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute KL divergence and JS distance between two datasets
        
        Args:
            df1: First dataset
            df2: Second dataset
            
        Returns:
            Dict containing KL divergence and JS distance
        """
        from scipy.stats import entropy
        from scipy.spatial.distance import jensenshannon
        
        # Select numeric features only
        numeric_cols = [col for col in df1.columns if col not in ['diagnosis', 'province']]
        
        # Compute histograms for each feature
        kl_divs = []
        js_dists = []
        
        for col in numeric_cols:
            # Create normalized histograms
            hist1, bins = np.histogram(df1[col], bins=30, density=True)
            hist2, _ = np.histogram(df2[col], bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            hist1 = hist1 + 1e-10
            hist2 = hist2 + 1e-10
            
            # Normalize
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # Compute KL divergence and JS distance
            kl_div = entropy(hist1, hist2)
            js_dist = jensenshannon(hist1, hist2)
            
            kl_divs.append(kl_div)
            js_dists.append(js_dist)
        
        return {
            'kl_divergence': np.mean(kl_divs),
            'js_distance': np.mean(js_dists)
        }


def generate_provincial_data(
    provinces: List[str] = ['ontario', 'alberta', 'quebec'],
    num_features: int = 10,
    seed: int = 42,
    save_path: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic data for all provinces
    
    Args:
        provinces: List of province names
        num_features: Number of clinical features
        seed: Random seed
        save_path: Path to save datasets (optional)
        
    Returns:
        Dict mapping province names to DataFrames
    """
    generator = ProvincialDataGenerator(seed=seed)
    datasets = {}
    
    for province in provinces:
        print(f"Generating data for {province.capitalize()}...")
        df = generator.generate_province_data(province, num_features=num_features)
        datasets[province] = df
        
        # Print statistics
        print(f"  - Samples: {len(df)}")
        print(f"  - Disease prevalence: {df['diagnosis'].mean():.3f}")
        print(f"  - Mean age: {df['age'].mean():.1f}")
        print(f"  - Female %: {df['sex'].mean()*100:.1f}%")
        
        # Save if path provided
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            file_path = f"{save_path}/{province}_cancer_data.csv"
            df.to_csv(file_path, index=False)
            print(f"  - Saved to: {file_path}")
    
    # Compute and print divergence metrics
    if len(provinces) >= 2:
        print("\nInter-provincial divergence metrics:")
        for i in range(len(provinces)):
            for j in range(i+1, len(provinces)):
                p1, p2 = provinces[i], provinces[j]
                metrics = generator.compute_divergence_metrics(
                    datasets[p1], datasets[p2]
                )
                print(f"  {p1.capitalize()}-{p2.capitalize()}: "
                      f"KL={metrics['kl_divergence']:.3f}, "
                      f"JS={metrics['js_distance']:.3f}")
    
    return datasets


def generate_cancer_datasets(seed: int = 42, save_path: str = None):
    """Generate datasets for cancer detection use case"""
    return generate_provincial_data(
        provinces=['ontario', 'alberta', 'quebec'],
        num_features=10,
        seed=seed,
        save_path=save_path
    )


def generate_pandemic_datasets(
    num_provinces: int = 5,
    num_features: int = 8,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic datasets for pandemic forecasting
    
    Args:
        num_provinces: Number of provinces to simulate
        num_features: Number of epidemiological features
        seed: Random seed
        
    Returns:
        Dict of provincial datasets
    """
    np.random.seed(seed)
    datasets = {}
    
    province_names = ['ontario', 'quebec', 'british_columbia', 'alberta', 'manitoba']
    
    for i, province in enumerate(province_names[:num_provinces]):
        n_samples = 2000 + i * 500  # Varying sample sizes
        
        # Generate time-series-like features
        time_steps = np.arange(n_samples)
        
        # ICU occupancy (with provincial variation)
        baseline_icu = 50 + i * 10
        seasonal_effect = 20 * np.sin(2 * np.pi * time_steps / 365)
        trend = 0.05 * time_steps
        noise = np.random.normal(0, 10, n_samples)
        icu_occupancy = baseline_icu + seasonal_effect + trend + noise
        icu_occupancy = np.clip(icu_occupancy, 0, 200)
        
        # Infection rate
        infection_rate = 0.05 + 0.02 * np.sin(2 * np.pi * time_steps / 180)
        infection_rate += np.random.normal(0, 0.01, n_samples)
        infection_rate = np.clip(infection_rate, 0, 0.15)
        
        # Other features
        data = {
            'time_step': time_steps,
            'icu_occupancy': icu_occupancy,
            'infection_rate': infection_rate,
            'hospitalization_rate': infection_rate * 0.15 + np.random.normal(0, 0.005, n_samples),
            'testing_rate': 0.1 + np.random.normal(0, 0.02, n_samples),
            'vaccination_rate': np.minimum(0.8, 0.002 * time_steps) + np.random.normal(0, 0.05, n_samples),
            'mortality_rate': infection_rate * 0.02 + np.random.normal(0, 0.001, n_samples),
            'province': province
        }
        
        # Target: ICU overload (binary classification)
        data['icu_overload'] = (icu_occupancy > 150).astype(int)
        
        datasets[province] = pd.DataFrame(data)
    
    return datasets


def generate_rare_disease_datasets(
    total_samples: int = 5000,
    num_features: int = 15,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic datasets for rare disease analysis
    
    Args:
        total_samples: Total number of rare disease cases
        num_features: Number of genetic/clinical features
        seed: Random seed
        
    Returns:
        Dict of institutional datasets
    """
    np.random.seed(seed)
    
    # Simulate multiple specialized centers
    institutions = ['sickkids_toronto', 'cheo_ottawa', 'bcch_vancouver']
    datasets = {}
    
    samples_per_inst = total_samples // len(institutions)
    
    for inst_id, inst_name in enumerate(institutions):
        # Generate genetic and clinical features
        features = {}
        
        for i in range(num_features):
            # Some features are institution-specific (non-IID)
            mean = 0.5 + inst_id * 0.1 * (i % 3)  # Introduce heterogeneity
            std = 0.2
            features[f'feature_{i}'] = np.random.normal(mean, std, samples_per_inst)
        
        # Generate rare disease phenotype labels (multi-class with heavy imbalance)
        # Phenotypes: 0=Type_A (common), 1=Type_B (rare), 2=Type_C (very rare)
        phenotype_probs = [0.70, 0.20, 0.10]  # Imbalanced distribution
        phenotypes = np.random.choice([0, 1, 2], size=samples_per_inst, p=phenotype_probs)
        
        data = features.copy()
        data['phenotype'] = phenotypes
        data['institution'] = inst_name
        
        datasets[inst_name] = pd.DataFrame(data)
    
    return datasets


def save_datasets_as_csv(datasets: Dict[str, pd.DataFrame], output_dir: str):
    """
    Save all datasets as CSV files
    
    Args:
        datasets: Dictionary of datasets
        output_dir: Output directory path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in datasets.items():
        filepath = os.path.join(output_dir, f"{name}_data.csv")
        df.to_csv(filepath, index=False)
        print(f"Saved {name} dataset to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Generating provincial cancer datasets...")
    datasets = generate_cancer_datasets(seed=42, save_path="./data_output")
    
    print("\nDataset statistics:")
    for province, df in datasets.items():
        print(f"\n{province.upper()}:")
        print(df.describe())
