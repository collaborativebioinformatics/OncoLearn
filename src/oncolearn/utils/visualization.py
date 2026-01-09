"""
Visualization utilities for training and model analysis.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_training_curves(
    metrics_tracker,
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Plot training curves.

    Args:
        metrics_tracker: MetricsTracker object
        metrics_to_plot: List of metric names to plot (None = all)
        save_path: Path to save figure
        figsize: Figure size
    """
    if metrics_to_plot is None:
        metrics_to_plot = list(metrics_tracker.metrics.keys())

    # Group metrics by prefix (train/val/test)
    train_metrics = [m for m in metrics_to_plot if m.startswith('train_')]
    val_metrics = [m for m in metrics_to_plot if m.startswith('val_')]

    num_plots = len(set([m.replace('train_', '').replace('val_', '')
                         for m in train_metrics + val_metrics]))

    fig, axes = plt.subplots(num_plots, 1, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0
    plotted_metrics = set()

    for metric in train_metrics:
        base_metric = metric.replace('train_', '')
        if base_metric in plotted_metrics:
            continue

        ax = axes[plot_idx]

        # Plot training
        train_history = metrics_tracker.get_history(metric)
        ax.plot(train_history, label=f'Train {base_metric}', linewidth=2)

        # Plot validation if available
        val_metric = f'val_{base_metric}'
        if val_metric in metrics_to_plot:
            val_history = metrics_tracker.get_history(val_metric)
            ax.plot(val_history, label=f'Val {base_metric}', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(base_metric)
        ax.set_title(f'{base_metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plotted_metrics.add(base_metric)
        plot_idx += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    normalize: bool = False
):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix [num_classes, num_classes]
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize by row
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    plt.figure(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names or list(range(cm.shape[1])),
        yticklabels=class_names or list(range(cm.shape[0])),
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def visualize_attention(
    attention_weights: torch.Tensor,
    image: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    num_heads: int = 8,
    layer_idx: int = -1
):
    """
    Visualize attention weights.

    Args:
        attention_weights: Attention weights [batch, num_heads, seq_len, seq_len]
                          or list of attention weights from multiple layers
        image: Optional input image to overlay attention on
        save_path: Path to save figure
        num_heads: Number of attention heads to visualize
        layer_idx: Which layer to visualize (if list of layers provided)
    """
    if isinstance(attention_weights, list):
        attention_weights = attention_weights[layer_idx]

    # Take first sample in batch
    if attention_weights.dim() == 4:
        attn = attention_weights[0]  # [num_heads, seq_len, seq_len]
    else:
        attn = attention_weights

    # Average over heads or show individual heads
    num_heads_to_show = min(num_heads, attn.shape[0])

    fig, axes = plt.subplots(2, num_heads_to_show // 2, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(num_heads_to_show):
        ax = axes[i]

        # Get attention for this head (typically from CLS token)
        head_attn = attn[i, 0, 1:].cpu().detach().numpy()  # Skip CLS token

        # Reshape to 2D if possible
        size = int(np.sqrt(len(head_attn)))
        if size * size == len(head_attn):
            head_attn = head_attn.reshape(size, size)

        im = ax.imshow(head_attn, cmap='viridis')
        ax.set_title(f'Head {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Attention Weights (from CLS token)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_loss_components(
    metrics_tracker,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Plot different loss components (classification, VAE, KLD, etc.).

    Args:
        metrics_tracker: MetricsTracker object
        save_path: Path to save figure
        figsize: Figure size
    """
    loss_components = ['total_loss', 'classification_loss', 'vae_loss',
                       'reconstruction_loss', 'kld_loss']

    available_losses = [
        f'train_{loss}' for loss in loss_components
        if f'train_{loss}' in metrics_tracker.metrics
    ]

    if not available_losses:
        print("No loss components found to plot")
        return

    fig, axes = plt.subplots(1, len(available_losses), figsize=figsize)
    if len(available_losses) == 1:
        axes = [axes]

    for i, loss_name in enumerate(available_losses):
        ax = axes[i]

        # Plot training loss
        train_history = metrics_tracker.get_history(loss_name)
        ax.plot(train_history, label='Train', linewidth=2, alpha=0.8)

        # Plot validation loss if available
        val_loss_name = loss_name.replace('train_', 'val_')
        if val_loss_name in metrics_tracker.metrics:
            val_history = metrics_tracker.get_history(val_loss_name)
            ax.plot(val_history, label='Val', linewidth=2, alpha=0.8)

        base_name = loss_name.replace('train_', '').replace('_', ' ').title()
        ax.set_title(base_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def visualize_latent_space(
    latent_features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    method: str = 'tsne',
    figsize: tuple = (10, 8)
):
    """
    Visualize latent space using dimensionality reduction.

    Args:
        latent_features: Latent features [N, latent_dim]
        labels: Labels [N]
        class_names: List of class names
        save_path: Path to save figure
        method: Dimensionality reduction method ('tsne', 'umap', 'pca')
        figsize: Figure size
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(latent_features)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced = reducer.fit_transform(latent_features)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(latent_features)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            reducer = TSNE(n_components=2, random_state=42)
            reduced = reducer.fit_transform(latent_features)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    plt.figure(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = class_names[label] if class_names else f'Class {label}'
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=[colors[i]],
            label=label_name,
            alpha=0.6,
            s=30
        )

    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Latent Space Visualization ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
