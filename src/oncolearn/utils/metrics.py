"""
Metrics computation and tracking utilities.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted logits [N, num_classes] or class indices [N]
        labels: Ground truth labels [N]
        num_classes: Number of classes
        prefix: Prefix for metric names

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if predictions.dim() == 2:
        preds = torch.argmax(predictions, dim=1).cpu().numpy()
        probs = torch.softmax(predictions, dim=1).cpu().numpy()
    else:
        preds = predictions.cpu().numpy()
        probs = None

    labels = labels.cpu().numpy()

    # Compute metrics
    metrics = {}

    # Accuracy
    accuracy = accuracy_score(labels, preds)
    metrics[f'{prefix}accuracy'] = accuracy

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    metrics[f'{prefix}precision'] = precision
    metrics[f'{prefix}recall'] = recall
    metrics[f'{prefix}f1'] = f1

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0)

    for i in range(min(len(precision_per_class), num_classes)):
        metrics[f'{prefix}precision_class_{i}'] = precision_per_class[i]
        metrics[f'{prefix}recall_class_{i}'] = recall_per_class[i]
        metrics[f'{prefix}f1_class_{i}'] = f1_per_class[i]

    # AUC-ROC (if probabilities available)
    if probs is not None and num_classes > 1:
        try:
            if num_classes == 2:
                auc = roc_auc_score(labels, probs[:, 1])
                metrics[f'{prefix}auc'] = auc
            else:
                auc = roc_auc_score(
                    labels, probs, multi_class='ovr', average='weighted')
                metrics[f'{prefix}auc'] = auc
        except ValueError:
            # Not all classes present in labels
            pass

    return metrics


class MetricsTracker:
    """
    Track metrics during training and validation.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(float)
        self.step_count = 0

    def update(self, metrics_dict: Dict[str, float], step: bool = True):
        """
        Update metrics.

        Args:
            metrics_dict: Dictionary of metric names to values
            step: Whether to increment step count
        """
        for name, value in metrics_dict.items():
            self.metrics[name].append(value)
            self.epoch_metrics[name] += value

        if step:
            self.step_count += 1

    def get_average(self, metric_name: str) -> float:
        """Get average value of a metric in current epoch."""
        if metric_name not in self.epoch_metrics:
            return 0.0
        return self.epoch_metrics[metric_name] / self.step_count

    def get_all_averages(self) -> Dict[str, float]:
        """Get average values of all metrics in current epoch."""
        if self.step_count == 0:
            return {}
        return {
            name: value / self.step_count
            for name, value in self.epoch_metrics.items()
        }

    def reset_epoch(self):
        """Reset epoch metrics and step count."""
        self.epoch_metrics = defaultdict(float)
        self.step_count = 0

    def get_history(self, metric_name: str) -> List[float]:
        """Get full history of a metric."""
        return self.metrics.get(metric_name, [])

    def get_best(self, metric_name: str, mode: str = 'max') -> float:
        """
        Get best value of a metric.

        Args:
            metric_name: Name of metric
            mode: 'max' or 'min'

        Returns:
            Best value
        """
        history = self.get_history(metric_name)
        if not history:
            return 0.0 if mode == 'max' else float('inf')

        if mode == 'max':
            return max(history)
        else:
            return min(history)

    def save(self, path: str):
        """Save metrics to file."""
        import json
        with open(path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)

    def load(self, path: str):
        """Load metrics from file."""
        import json
        with open(path, 'r') as f:
            self.metrics = defaultdict(list, json.load(f))


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title to print
    """
    print(f"\n{title}")
    print("-" * 50)
    for name, value in metrics.items():
        print(f"{name:30s}: {value:.4f}")
    print("-" * 50)


def get_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted logits or class indices
        labels: Ground truth labels
        num_classes: Number of classes

    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    if predictions.dim() == 2:
        preds = torch.argmax(predictions, dim=1).cpu().numpy()
    else:
        preds = predictions.cpu().numpy()

    labels = labels.cpu().numpy()

    return confusion_matrix(labels, preds, labels=list(range(num_classes)))


def get_classification_report(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Generate classification report.

    Args:
        predictions: Predicted logits or class indices
        labels: Ground truth labels
        class_names: Optional list of class names

    Returns:
        Classification report string
    """
    if predictions.dim() == 2:
        preds = torch.argmax(predictions, dim=1).cpu().numpy()
    else:
        preds = predictions.cpu().numpy()

    labels = labels.cpu().numpy()

    return classification_report(labels, preds, target_names=class_names)
