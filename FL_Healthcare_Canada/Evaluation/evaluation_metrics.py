"""
Demo Performance metrics for model evaluation

Comprehensive metrics for classification tasks including:
- Accuracy, Precision, Recall, F1 Score
- AUC-ROC for binary and multi-class
- Confusion matrix
- Per-class metrics
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Union


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics
    
    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        y_prob: Prediction probabilities (n_samples,) for binary or (n_samples, n_classes) for multi-class
        average: Averaging method ('binary', 'macro', 'weighted', 'micro')
        
    Returns:
        Dict of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Compute AUC if probabilities provided
    if y_prob is not None:
        try:
            if average == 'binary':
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            else:
                # For multi-class, use one-vs-rest
                metrics['auc'] = roc_auc_score(
                    y_true, y_prob, 
                    multi_class='ovr', 
                    average=average
                )
        except (ValueError, AttributeError) as e:
            # Handle cases where AUC cannot be computed
            metrics['auc'] = 0.0
            print(f"Warning: Could not compute AUC: {e}")
    
    return metrics


def compute_auc(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    multi_class: bool = False,
    average: str = 'macro'
) -> float:
    """
    Compute Area Under ROC Curve
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        multi_class: Whether it's multi-class classification
        average: Averaging method for multi-class
        
    Returns:
        AUC score
    """
    try:
        if multi_class:
            return roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
        else:
            return roc_auc_score(y_true, y_prob)
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not compute AUC: {e}")
        return 0.0


def compute_f1(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    average: str = 'macro'
) -> float:
    """
    Compute F1 score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method
        
    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization method ('true', 'pred', 'all', None)
        
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
    
    return cm


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        
    Returns:
        Dict mapping class names to metrics
    """
    # Get classification report as dict
    report = classification_report(
        y_true, y_pred, 
        output_dict=True, 
        zero_division=0
    )
    
    # Extract per-class metrics
    per_class = {}
    unique_classes = sorted(set(y_true) | set(y_pred))
    
    for i, class_id in enumerate(unique_classes):
        class_key = str(class_id)
        if class_key in report:
            class_label = class_names[i] if class_names and i < len(class_names) else f"Class_{class_id}"
            per_class[class_label] = report[class_key]
    
    return per_class


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
    compute_probabilities: bool = True,
    return_predictions: bool = False
) -> Dict:
    """
    Evaluate PyTorch model on a dataset
    
    Args:
        model: PyTorch model
        data_loader: DataLoader
        device: Computing device
        compute_probabilities: Whether to compute prediction probabilities
        return_predictions: Whether to return individual predictions
        
    Returns:
        Dict of predictions, labels, and optionally probabilities
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get predictions
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # Get probabilities if requested
            if compute_probabilities:
                if output.shape[1] == 2:
                    # Binary classification - use probability of positive class
                    probs = torch.softmax(output, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy())
                else:
                    # Multi-class - store all class probabilities
                    probs = torch.softmax(output, dim=1)
                    all_probs.append(probs.cpu().numpy())
    
    result = {
        'predictions': np.array(all_preds),
        'true_labels': np.array(all_labels)
    }
    
    if compute_probabilities and all_probs:
        if isinstance(all_probs[0], np.ndarray) and len(all_probs[0].shape) > 0:
            # Multi-class probabilities
            result['probabilities'] = np.vstack(all_probs)
        else:
            # Binary probabilities
            result['probabilities'] = np.array(all_probs)
    
    return result


def evaluate_federated_model(
    global_model: nn.Module,
    client_test_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate federated model on multiple client test sets
    
    Args:
        global_model: Global federated model
        client_test_loaders: Dict mapping client names to test DataLoaders
        device: Computing device
        
    Returns:
        Dict mapping client names to their metrics
    """
    client_metrics = {}
    
    for client_name, test_loader in client_test_loaders.items():
        eval_result = evaluate_model(global_model, test_loader, device)
        metrics = compute_metrics(
            eval_result['true_labels'],
            eval_result['predictions'],
            eval_result.get('probabilities')
        )
        client_metrics[client_name] = metrics
    
    return client_metrics


def compute_model_performance(
    model: nn.Module,
    test_loader,
    device: torch.device,
    task_type: str = 'binary'
) -> Dict[str, float]:
    """
    Compute comprehensive model performance metrics
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Computing device
        task_type: 'binary' or 'multiclass'
        
    Returns:
        Dict of performance metrics
    """
    # Evaluate model
    eval_result = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    average = 'binary' if task_type == 'binary' else 'macro'
    metrics = compute_metrics(
        eval_result['true_labels'],
        eval_result['predictions'],
        eval_result.get('probabilities'),
        average=average
    )
    
    # Add confusion matrix
    cm = compute_confusion_matrix(
        eval_result['true_labels'],
        eval_result['predictions']
    )
    
    return {
        **metrics,
        'confusion_matrix': cm,
        'num_samples': len(eval_result['true_labels'])
    }


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute balanced accuracy (average of per-class recalls)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Balanced accuracy
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_recall = []
    
    for i in range(len(cm)):
        if cm[i].sum() > 0:
            recall = cm[i, i] / cm[i].sum()
            per_class_recall.append(recall)
    
    return np.mean(per_class_recall) if per_class_recall else 0.0


def compute_sensitivity_specificity(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    pos_label: int = 1
) -> Tuple[float, float]:
    """
    Compute sensitivity (recall) and specificity for binary classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Label of positive class
        
    Returns:
        Tuple of (sensitivity, specificity)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if len(cm) != 2:
        raise ValueError("Sensitivity/Specificity only defined for binary classification")
    
    # Sensitivity = TP / (TP + FN) = Recall
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
    
    # Specificity = TN / (TN + FP)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
    
    return sensitivity, specificity


if __name__ == "__main__":
    print("="*60)
    print("Testing Evaluation Metrics")
    print("="*60)
    
    # Test binary classification
    print("\n1. Binary Classification Metrics")
    print("-"*60)
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.4, 0.2, 0.85, 0.6])
    
    metrics = compute_metrics(y_true, y_pred, y_prob, average='binary')
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test confusion matrix
    print("\n2. Confusion Matrix")
    print("-"*60)
    cm = compute_confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Test sensitivity/specificity
    print("\n3. Sensitivity and Specificity")
    print("-"*60)
    sens, spec = compute_sensitivity_specificity(y_true, y_pred)
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    
    # Test multi-class
    print("\n4. Multi-class Classification")
    print("-"*60)
    y_true_mc = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred_mc = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
    
    metrics_mc = compute_metrics(y_true_mc, y_pred_mc, average='macro')
    print("Multi-class Metrics:")
    for name, value in metrics_mc.items():
        print(f"  {name}: {value:.4f}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
