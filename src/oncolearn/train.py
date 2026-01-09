"""
Standalone training script for cancer subtyping model.
YOLO → Attention → VAE → ViT Classifier pipeline.

Usage:
    python train_standalone.py --config config/train_config.yaml
    python train_standalone.py --data_dir data/TCIA --clinical_file data/GDCdata/TCGA-BRCA.clinical.tsv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from oncolearn.models import CancerSubtypingModel
from oncolearn.utils.config import Config
from oncolearn.utils.data_loader import create_data_loaders
from oncolearn.utils.metrics import MetricsTracker, compute_metrics
from oncolearn.utils.visualization import plot_training_curves

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class StandaloneTrainer:
    """Trainer for standalone (non-federated) training."""

    def __init__(self, config: Config):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Set random seed
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Create data loaders
        print("\nLoading data...")
        self.train_loader, self.val_loader, self.test_loader, num_classes = create_data_loaders(
            data_dir=config.data_dir,
            clinical_file=config.clinical_file,
            genetic_data_dir=getattr(
                config, 'genetic_data_dir', '/workspace/data/processed'),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            seed=config.random_seed,
            label_column=getattr(config, 'label_column',
                                 'ajcc_pathologic_stage.diagnoses'),
            use_genetic_data=getattr(config, 'use_genetic_data', True),
            max_genes=getattr(config, 'max_genes', 1000),
            cancer_type=getattr(config, 'cancer_type', None),
            extension=getattr(config, 'image_extension', '*.png'),
            normalize_genes=getattr(config, 'normalize_genes', True),
            standardize_genes=getattr(config, 'standardize_genes', True),
        )

        # Update config with detected num_classes
        config.num_classes = max(num_classes, config.num_classes)
        print(f"Number of classes: {config.num_classes}")

        # Create model
        print("\nInitializing model...")
        self.model = CancerSubtypingModel(
            yolo_model=config.yolo_model_name,
            freeze_yolo=config.freeze_yolo,
            num_attention_layers=config.num_attention_layers,
            num_attention_heads=config.num_attention_heads,
            latent_dim=config.latent_dim,
            num_classes=config.num_classes,
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Create learning rate scheduler
        if config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs
            )
        elif config.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.num_epochs // 3,
                gamma=0.1
            )
        elif config.scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config.scheduler_patience,
                factor=0.5
            )
        else:
            self.scheduler = None

        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        # Save config
        config.save(self.output_dir / "config.yaml")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        # Set training mode manually to avoid YOLO train() method conflict
        # Recursively set training flag for all modules except YOLO
        self.model.training = True
        for name, module in self.model.named_modules():
            # Skip YOLO and its submodules
            if 'yolo' in name.lower() or module.__class__.__module__.startswith('ultralytics'):
                continue
            module.training = True

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        # Initialize tracking variables
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images, labels=labels)
                    loss = outputs['loss']

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, labels=labels)
                loss = outputs['loss']

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )

                self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            all_predictions.append(outputs['logits'].detach().cpu())
            all_labels.append(labels.cpu())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_metrics(
            all_predictions,
            all_labels,
            num_classes=self.config.num_classes,
            prefix="train_"
        )
        metrics['train_loss'] = avg_loss

        return metrics

    def validate(self, epoch: int, loader_name: str = "val") -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number
            loader_name: Name of loader ("val" or "test")

        Returns:
            Dictionary of validation metrics
        """
        loader = self.val_loader if loader_name == "val" else self.test_loader

        # Set eval mode manually to avoid YOLO train() method conflict
        self.model.training = False
        for module in self.model.modules():
            module.training = False

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{loader_name.capitalize()} evaluation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(images, labels=labels)
                        loss = outputs['loss']
                else:
                    outputs = self.model(images, labels=labels)
                    loss = outputs['loss']

                total_loss += loss.item()
                all_predictions.append(outputs['logits'].cpu())
                all_labels.append(labels.cpu())

        # Compute metrics
        avg_loss = total_loss / len(loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_metrics(
            all_predictions,
            all_labels,
            num_classes=self.config.num_classes,
            prefix=f"{loader_name}_"
        )
        metrics[f'{loader_name}_loss'] = avg_loss

        return metrics

    def train(self):
        """Run full training loop."""
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60 + "\n")

        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch(epoch)
            self.metrics_tracker.update(train_metrics)

            # Validate
            val_metrics = self.validate(epoch, loader_name="val")
            self.metrics_tracker.update(val_metrics)

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Train Acc: {train_metrics['train_accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Acc: {val_metrics['val_accuracy']:.4f}")

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # Save best model
            if val_metrics['val_accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_accuracy']
                self.best_val_loss = val_metrics['val_loss']
                self.epochs_without_improvement = 0

                checkpoint_path = self.output_dir / "best_model.pth"
                self.save_checkpoint(checkpoint_path, epoch, is_best=True)
                print(
                    f"  ✓ New best model saved (acc={self.best_val_acc:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = self.output_dir / \
                    f"checkpoint_epoch_{epoch+1}.pth"
                self.save_checkpoint(checkpoint_path, epoch)

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Final evaluation on test set
        print("\n" + "="*60)
        print("Final evaluation on test set...")
        print("="*60 + "\n")

        # Load best model
        best_checkpoint = self.output_dir / "best_model.pth"
        if best_checkpoint.exists():
            self.load_checkpoint(best_checkpoint)

        test_metrics = self.validate(0, loader_name="test")

        print("\nTest Set Results:")
        print(f"  Test Loss: {test_metrics['test_loss']:.4f}")
        print(f"  Test Acc: {test_metrics['test_accuracy']:.4f}")
        print(f"  Test F1: {test_metrics['test_f1']:.4f}")

        # Save metrics
        self.metrics_tracker.save_history(
            self.output_dir / "metrics_history.json")

        # Plot training curves
        plot_training_curves(
            self.metrics_tracker.metrics,
            save_path=self.output_dir / "training_curves.png"
        )

        print(f"\n✓ Training complete! Results saved to {self.output_dir}")

    def save_checkpoint(self, path: Path, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config.to_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)

        print(f"Loaded checkpoint from {path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train cancer subtyping model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--clinical_file", type=str,
                        help="Path to clinical data file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        config = Config.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = Config()
        print("Using default config")

    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.clinical_file:
        config.clinical_file = args.clinical_file
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.device:
        config.device = args.device

    # Create trainer and train
    trainer = StandaloneTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
