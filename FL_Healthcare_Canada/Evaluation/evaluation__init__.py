"""
Demo Evaluation module for Federated Learning Healthcare

Includes:
- Performance metrics (accuracy, AUC, F1, etc.)
- Statistical significance tests (t-tests, CIs)
- Visualization utilities (confusion matrices, plots)
"""

from .metrics import (
    compute_metrics,
    compute_auc,
    compute_f1,
    compute_confusion_matrix,
    compute_per_class_metrics,
    evaluate_model,
    evaluate_federated_model,
    compute_model_performance,
    compute_balanced_accuracy,
    compute_sensitivity_specificity
)

from .statistical_tests import (
    paired_t_test,
    independent_t_test,
    compute_confidence_interval,
    compute_summary_statistics,
    wilcoxon_signed_rank_test,
    mann_whitney_u_test,
    bonferroni_correction,
    holm_bonferroni_correction,
    compute_cohens_d,
    check_normality,
    format_results_table
)

from .visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_comparison_bars,
    plot_comparison_boxplots,
    plot_roc_curve,
    plot_federated_client_comparison,
    plot_convergence_comparison
)

__all__ = [
    # Metrics
    'compute_metrics',
    'compute_auc',
    'compute_f1',
    'compute_confusion_matrix',
    'compute_per_class_metrics',
    'evaluate_model',
    'evaluate_federated_model',
    'compute_model_performance',
    'compute_balanced_accuracy',
    'compute_sensitivity_specificity',
    
    # Statistical Tests
    'paired_t_test',
    'independent_t_test',
    'compute_confidence_interval',
    'compute_summary_statistics',
    'wilcoxon_signed_rank_test',
    'mann_whitney_u_test',
    'bonferroni_correction',
    'holm_bonferroni_correction',
    'compute_cohens_d',
    'check_normality',
    'format_results_table',
    
    # Visualization
    'plot_confusion_matrix',
    'plot_training_curves',
    'plot_comparison_bars',
    'plot_comparison_boxplots',
    'plot_roc_curve',
    'plot_federated_client_comparison',
    'plot_convergence_comparison'
]

__version__ = '1.0.0'
__author__ = 'Abbas Yazdinejad and Jude Kong'
