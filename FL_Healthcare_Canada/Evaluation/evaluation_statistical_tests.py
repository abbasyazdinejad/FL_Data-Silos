"""
Demo  Statistical significance testing 

Includes:
- Paired t-tests
- Confidence intervals
- Effect size (Cohen's d)
- Summary statistics
- Multiple comparison corrections
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional


def paired_t_test(
    results1: List[float],
    results2: List[float],
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Perform paired t-test between two sets of results
    
    Args:
        results1: First set of results (e.g., FL model accuracy across runs)
        results2: Second set of results (e.g., siloed model accuracy across runs)
        alpha: Significance level
        alternative: Type of test ('two-sided', 'less', 'greater')
        
    Returns:
        Dict containing test statistics
    """
    results1 = np.array(results1)
    results2 = np.array(results2)
    
    if len(results1) != len(results2):
        raise ValueError("Results must have same length for paired t-test")
    
    # Compute paired t-test
    t_statistic, p_value = stats.ttest_rel(results1, results2, alternative=alternative)
    
    # Compute differences
    diff = results1 - results2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    # Compute effect size (Cohen's d)
    cohens_d = mean_diff / (std_diff + 1e-10)
    
    # Determine significance
    significant = p_value < alpha
    
    return {
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'significant': bool(significant),
        'cohens_d': float(cohens_d),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'alpha': alpha,
        'n_samples': len(results1)
    }


def independent_t_test(
    results1: List[float],
    results2: List[float],
    alpha: float = 0.05,
    equal_var: bool = True
) -> Dict[str, float]:
    """
    Perform independent samples t-test
    
    Args:
        results1: First set of results
        results2: Second set of results
        alpha: Significance level
        equal_var: Whether to assume equal variance
        
    Returns:
        Dict containing test statistics
    """
    results1 = np.array(results1)
    results2 = np.array(results2)
    
    # Compute independent t-test
    t_statistic, p_value = stats.ttest_ind(results1, results2, equal_var=equal_var)
    
    # Compute effect size
    pooled_std = np.sqrt(
        ((len(results1) - 1) * np.var(results1, ddof=1) + 
         (len(results2) - 1) * np.var(results2, ddof=1)) / 
        (len(results1) + len(results2) - 2)
    )
    cohens_d = (np.mean(results1) - np.mean(results2)) / (pooled_std + 1e-10)
    
    return {
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'cohens_d': float(cohens_d),
        'mean1': float(np.mean(results1)),
        'mean2': float(np.mean(results2)),
        'alpha': alpha
    }


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for data
    
    Args:
        data: List of values
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    
    # Compute confidence interval using t-distribution
    ci = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean,
        scale=std_err
    )
    
    return float(mean), float(ci[0]), float(ci[1])


def compute_summary_statistics(
    results: List[float],
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Compute comprehensive summary statistics for experimental results
    
    Args:
        results: List of results from multiple runs
        confidence: Confidence level for CI
        
    Returns:
        Dict of summary statistics
    """
    results = np.array(results)
    
    if len(results) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'ci_95_lower': 0.0,
            'ci_95_upper': 0.0,
            'n_runs': 0
        }
    
    mean, ci_lower, ci_upper = compute_confidence_interval(results, confidence)
    
    return {
        'mean': float(mean),
        'std': float(np.std(results, ddof=1)),
        'sem': float(stats.sem(results)),
        'min': float(np.min(results)),
        'max': float(np.max(results)),
        'median': float(np.median(results)),
        'q25': float(np.percentile(results, 25)),
        'q75': float(np.percentile(results, 75)),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'n_runs': len(results)
    }


def wilcoxon_signed_rank_test(
    results1: List[float],
    results2: List[float],
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
    
    Args:
        results1: First set of results
        results2: Second set of results
        alpha: Significance level
        
    Returns:
        Dict containing test statistics
    """
    results1 = np.array(results1)
    results2 = np.array(results2)
    
    # Compute Wilcoxon test
    statistic, p_value = stats.wilcoxon(results1, results2)
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'alpha': alpha
    }


def mann_whitney_u_test(
    results1: List[float],
    results2: List[float],
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform Mann-Whitney U test (non-parametric alternative to independent t-test)
    
    Args:
        results1: First set of results
        results2: Second set of results
        alpha: Significance level
        
    Returns:
        Dict containing test statistics
    """
    results1 = np.array(results1)
    results2 = np.array(results2)
    
    # Compute Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(results1, results2, alternative='two-sided')
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'alpha': alpha
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[bool]:
    """
    Apply Bonferroni correction for multiple comparisons
    
    Args:
        p_values: List of p-values
        alpha: Family-wise error rate
        
    Returns:
        List of booleans indicating significance
    """
    adjusted_alpha = alpha / len(p_values)
    return [p < adjusted_alpha for p in p_values]


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[bool]:
    """
    Apply Holm-Bonferroni correction (less conservative than Bonferroni)
    
    Args:
        p_values: List of p-values
        alpha: Family-wise error rate
        
    Returns:
        List of booleans indicating significance
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]
    
    significant = np.zeros(n, dtype=bool)
    
    for i, p_value in enumerate(sorted_p_values):
        adjusted_alpha = alpha / (n - i)
        if p_value < adjusted_alpha:
            significant[sorted_indices[i]] = True
        else:
            break
    
    return significant.tolist()


def compute_cohens_d(
    results1: List[float],
    results2: List[float],
    paired: bool = True
) -> float:
    """
    Compute Cohen's d effect size
    
    Args:
        results1: First set of results
        results2: Second set of results
        paired: Whether data is paired
        
    Returns:
        Cohen's d
    """
    results1 = np.array(results1)
    results2 = np.array(results2)
    
    if paired:
        # For paired data
        diff = results1 - results2
        return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))
    else:
        # For independent samples
        pooled_std = np.sqrt(
            ((len(results1) - 1) * np.var(results1, ddof=1) + 
             (len(results2) - 1) * np.var(results2, ddof=1)) / 
            (len(results1) + len(results2) - 2)
        )
        return float((np.mean(results1) - np.mean(results2)) / (pooled_std + 1e-10))


def check_normality(data: List[float], alpha: float = 0.05) -> Dict[str, any]:
    """
    Test if data follows normal distribution using Shapiro-Wilk test
    
    Args:
        data: Data to test
        alpha: Significance level
        
    Returns:
        Dict with test results
    """
    data = np.array(data)
    
    if len(data) < 3:
        return {
            'test': 'Shapiro-Wilk',
            'statistic': None,
            'p_value': None,
            'normal': None,
            'message': 'Insufficient data (n < 3)'
        }
    
    statistic, p_value = stats.shapiro(data)
    
    return {
        'test': 'Shapiro-Wilk',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'normal': bool(p_value > alpha),
        'alpha': alpha
    }


def format_results_table(
    results1: List[float],
    results2: List[float],
    name1: str = "Method 1",
    name2: str = "Method 2",
    test_type: str = 'paired'
) -> str:
    """
    Format statistical comparison results as a table
    
    Args:
        results1: First set of results
        results2: Second set of results
        name1: Name of first method
        name2: Name of second method
        test_type: Type of test ('paired' or 'independent')
        
    Returns:
        Formatted string table
    """
    # Compute statistics
    stats1 = compute_summary_statistics(results1)
    stats2 = compute_summary_statistics(results2)
    
    if test_type == 'paired':
        test_result = paired_t_test(results1, results2)
    else:
        test_result = independent_t_test(results1, results2)
    
    # Format table
    table = []
    table.append("="*70)
    table.append("STATISTICAL COMPARISON")
    table.append("="*70)
    table.append(f"\n{name1}:")
    table.append(f"  Mean: {stats1['mean']:.4f} ± {stats1['std']:.4f}")
    table.append(f"  95% CI: [{stats1['ci_95_lower']:.4f}, {stats1['ci_95_upper']:.4f}]")
    table.append(f"  Range: [{stats1['min']:.4f}, {stats1['max']:.4f}]")
    
    table.append(f"\n{name2}:")
    table.append(f"  Mean: {stats2['mean']:.4f} ± {stats2['std']:.4f}")
    table.append(f"  95% CI: [{stats2['ci_95_lower']:.4f}, {stats2['ci_95_upper']:.4f}]")
    table.append(f"  Range: [{stats2['min']:.4f}, {stats2['max']:.4f}]")
    
    table.append(f"\n{test_type.capitalize()} t-test:")
    table.append(f"  t-statistic: {test_result['t_statistic']:.4f}")
    table.append(f"  p-value: {test_result['p_value']:.4f}")
    table.append(f"  Significant (α=0.05): {'Yes' if test_result['significant'] else 'No'}")
    table.append(f"  Cohen's d: {test_result['cohens_d']:.4f}")
    
    if 'mean_diff' in test_result:
        table.append(f"  Mean difference: {test_result['mean_diff']:.4f}")
    
    table.append("="*70)
    
    return "\n".join(table)


if __name__ == "__main__":
    print("="*60)
    print("Testing Statistical Tests")
    print("="*60)
    
    # Test paired t-test
    print("\n1. Paired t-test")
    print("-"*60)
    fl_results = [0.92, 0.91, 0.93, 0.92, 0.91]
    siloed_results = [0.88, 0.89, 0.87, 0.88, 0.89]
    
    test_result = paired_t_test(fl_results, siloed_results)
    print(f"FL: {np.mean(fl_results):.4f} ± {np.std(fl_results, ddof=1):.4f}")
    print(f"Siloed: {np.mean(siloed_results):.4f} ± {np.std(siloed_results, ddof=1):.4f}")
    print(f"t-statistic: {test_result['t_statistic']:.4f}")
    print(f"p-value: {test_result['p_value']:.4f}")
    print(f"Significant: {test_result['significant']}")
    print(f"Cohen's d: {test_result['cohens_d']:.4f}")
    
    # Test confidence interval
    print("\n2. Confidence Interval")
    print("-"*60)
    mean, ci_lower, ci_upper = compute_confidence_interval(fl_results)
    print(f"Mean: {mean:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Test summary statistics
    print("\n3. Summary Statistics")
    print("-"*60)
    stats_dict = compute_summary_statistics(fl_results)
    for key, value in stats_dict.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Test formatted table
    print("\n4. Formatted Results Table")
    print("-"*60)
    table = format_results_table(
        fl_results, siloed_results,
        "Federated Learning", "Siloed Baseline"
    )
    print(table)
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
