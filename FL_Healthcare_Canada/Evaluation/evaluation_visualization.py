"""
Visualization utilities 

Includes:
- Confusion matrix plots
- Training curves (loss, accuracy)
- ROC curves
- Box plots for comparisons
- Bar charts for metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot confusion matrix with annotations
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure (optional)
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        cmap: Color map
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
        cm_display = cm * 100  # Convert to percentage for display
    else:
        fmt = 'd'
        cm_display = cm
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        square=True
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
    metrics: Optional[List[str]] = None
):
    """
    Plot training curves (loss, accuracy, etc.)
    
    Args:
        history: Dict containing metric histories
        save_path: Path to save figure
        title: Plot title
        metrics: List of metrics to plot (if None, plots all)
    """
    if metrics is None:
        metrics = list(history.keys())
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            values = history[metric]
            ax.plot(values, linewidth=2, marker='o', markersize=4)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at best value
            if 'loss' not in metric.lower():
                best_val = max(values)
                best_round = values.index(best_val)
                ax.axhline(y=best_val, color='r', linestyle='--', alpha=0.5, 
                          label=f'Best: {best_val:.4f} (Round {best_round})')
                ax.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.close()


def plot_comparison_bars(
    results: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Performance Comparison",
    ylabel: str = "Metric Value",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot bar chart comparing different methods/metrics
    
    Args:
        results: Dict mapping method names to metric values
        save_path: Path to save figure
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    methods = list(results.keys())
    values = list(results.values())
    
    colors = sns.color_palette("husl", len(methods))
    bars = plt.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison bars to {save_path}")
    
    plt.close()


def plot_comparison_boxplots(
    data_dict: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Performance Comparison",
    ylabel: str = "Metric Value",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot box plots comparing distributions across multiple runs
    
    Args:
        data_dict: Dict mapping method names to lists of values
        save_path: Path to save figure
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    methods = list(data_dict.keys())
    data = [data_dict[method] for method in methods]
    
    bp = plt.boxplot(data, labels=methods, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Color boxes
    colors = sns.color_palette("Set2", len(methods))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved boxplots to {save_path}")
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 8)
):
    """
    Plot ROC curve for binary classification
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=figsize)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    plt.close()


def plot_federated_client_comparison(
    client_metrics: Dict[str, Dict[str, float]],
    metric_name: str = 'accuracy',
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot comparison of a specific metric across federated clients
    
    Args:
        client_metrics: Dict mapping client names to their metrics dicts
        metric_name: Name of metric to plot
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    clients = list(client_metrics.keys())
    values = [client_metrics[client][metric_name] for client in clients]
    
    colors = sns.color_palette("Set3", len(clients))
    bars = plt.bar(clients, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add average line
    avg_value = np.mean(values)
    plt.axhline(y=avg_value, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_value:.3f}')
    
    if title is None:
        title = f'{metric_name.title()} Across Federated Clients'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(metric_name.title(), fontsize=12)
    plt.xlabel('Client', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved client comparison to {save_path}")
    
    plt.close()


def plot_convergence_comparison(
    histories: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Convergence Comparison",
    ylabel: str = "Accuracy",
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot convergence curves for multiple methods
    
    Args:
        histories: Dict mapping method names to metric histories
        save_path: Path to save figure
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for method_name, values in histories.items():
        plt.plot(values, marker='o', markersize=4, linewidth=2, 
                label=method_name, alpha=0.8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence comparison to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("Testing Visualization Functions")
    print("="*60)
    
    # Create output directory
    os.makedirs("test_plots", exist_ok=True)
    
    # Test confusion matrix
    print("\n1. Confusion Matrix")
    print("-"*60)
    cm = np.array([[85, 5, 3, 2],
                   [4, 90, 4, 2],
                   [3, 5, 87, 5],
                   [2, 3, 4, 91]])
    class_names = ['Class A', 'Class B', 'Class C', 'Class D']
    plot_confusion_matrix(cm, class_names, 
                         save_path="test_plots/confusion_matrix.png",
                         title="Test Confusion Matrix")
    
    # Test training curves
    print("\n2. Training Curves")
    print("-"*60)
    history = {
        'loss': [0.8, 0.6, 0.5, 0.4, 0.35, 0.32, 0.30, 0.29, 0.28, 0.27],
        'accuracy': [0.75, 0.82, 0.85, 0.88, 0.90, 0.91, 0.92, 0.92, 0.93, 0.93]
    }
    plot_training_curves(history, 
                        save_path="test_plots/training_curves.png",
                        title="Test Training Curves")
    
    # Test comparison bars
    print("\n3. Comparison Bars")
    print("-"*60)
    results = {
        'Federated': 0.92,
        'Siloed': 0.88,
        'Centralized': 0.93,
        'Baseline': 0.75
    }
    plot_comparison_bars(results, 
                        save_path="test_plots/comparison_bars.png",
                        title="Performance Comparison",
                        ylabel="Accuracy")
    
    # Test boxplots
    print("\n4. Boxplots")
    print("-"*60)
    data_dict = {
        'FL': [0.91, 0.92, 0.93, 0.92, 0.91],
        'Siloed': [0.87, 0.88, 0.89, 0.88, 0.87],
        'Centralized': [0.93, 0.94, 0.93, 0.94, 0.93]
    }
    plot_comparison_boxplots(data_dict, 
                            save_path="test_plots/boxplots.png",
                            title="Distribution Comparison",
                            ylabel="Accuracy")
    
    print("\n" + "="*60)
    print("All plots saved to test_plots/")
    print("="*60)
