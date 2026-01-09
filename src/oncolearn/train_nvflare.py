"""
NVFlare federated learning training script for cancer subtyping model.
YOLO → Attention → VAE → ViT Classifier pipeline with federated learning.

This script uses NVFlare's Client API for federated learning.

Usage:
    python train_nvflare.py --data_dir /data/site1 --batch_size 8 --local_epochs 2
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# NVFlare Client API
import nvflare.client as flare
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from oncolearn.models import CancerSubtypingModel
from oncolearn.utils.data_loader import create_data_loaders
from oncolearn.utils.metrics import compute_metrics


class NVFlareTrainer:
    """Trainer for federated learning with NVFlare."""

    def __init__(
        self,
        data_dir: str,
        clinical_file: Optional[str] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        local_epochs: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        image_size: tuple = (512, 512),
        num_classes: int = 5,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_clip_val: float = 1.0,
        model_path: Optional[str] = None,
    ):
        """
        Initialize NVFlare trainer.

        Args:
            data_dir: Directory containing local training data
            clinical_file: Path to clinical data file
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            local_epochs: Number of local training epochs per round
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            image_size: Input image size
            num_classes: Number of output classes
            device: Device to use (cuda/cpu)
            mixed_precision: Whether to use mixed precision training
            gradient_clip_val: Gradient clipping value
            model_path: Path to save best local model
        """
        self.data_dir = data_dir
        self.clinical_file = clinical_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.image_size = image_size
        self.num_classes = num_classes
        self.gradient_clip_val = gradient_clip_val
        self.model_path = model_path or f"{data_dir}/best_model.pth"
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Mixed precision
        self.scaler = GradScaler() if mixed_precision and torch.cuda.is_available() else None

        # Load data
        print(f"\nLoading data from {data_dir}...")
        self.train_loader, self.val_loader, _, detected_classes = create_data_loaders(
            data_dir=data_dir,
            clinical_file=clinical_file,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            train_split=0.9,  # Use 90% for training in federated setting
            val_split=0.1,   # 10% for local validation
        )
        
        # Update num_classes if detected
        if detected_classes > 0:
            self.num_classes = detected_classes

        # Initialize model
        print("\nInitializing model...")
        self.model = CancerSubtypingModel(
            yolo_model="yolov8n.pt",
            freeze_yolo=True,
            num_attention_layers=2,
            num_attention_heads=8,
            latent_dim=128,
            num_classes=self.num_classes,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Best local accuracy tracking
        self.best_local_accuracy = 0.0

    def train_local(self, model_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform local training for specified number of epochs.

        Args:
            model_weights: Model weights from server

        Returns:
            Updated model weights
        """
        # Load weights from server
        self.model.load_state_dict(model_weights)
        self.model.train()

        print(f"\nTraining for {self.local_epochs} local epochs...")
        
        for epoch in range(self.local_epochs):
            total_loss = 0.0
            all_predictions = []
            all_labels = []

            pbar = tqdm(
                self.train_loader,
                desc=f"Local Epoch {epoch+1}/{self.local_epochs}"
            )

            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(images, labels=labels)
                        loss = outputs['loss']
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_val
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images, labels=labels)
                    loss = outputs['loss']
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_val
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
                num_classes=self.num_classes
            )
            
            print(f"  Loss: {avg_loss:.4f}, Acc: {metrics['accuracy']:.4f}")

        # Return updated weights
        return self.model.state_dict()

    def evaluate_local(self, model_weights: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate model on local validation set.

        Args:
            model_weights: Model weights to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        # Load weights
        self.model.load_state_dict(model_weights)
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Local validation"):
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
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        metrics = compute_metrics(
            all_predictions,
            all_labels,
            num_classes=self.num_classes
        )
        metrics['loss'] = avg_loss

        return metrics

    def save_best_model(self, model_weights: Dict[str, torch.Tensor], accuracy: float):
        """
        Save model if it achieves best local accuracy.

        Args:
            model_weights: Model weights to save
            accuracy: Current accuracy
        """
        if accuracy > self.best_local_accuracy:
            self.best_local_accuracy = accuracy
            torch.save(model_weights, self.model_path)
            print(f"✓ Saved best local model (acc={accuracy:.4f}) to {self.model_path}")


def main():
    """Main federated training function."""
    parser = argparse.ArgumentParser(description="NVFlare federated training")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to local data directory")
    parser.add_argument("--clinical_file", type=str, default=None,
                       help="Path to clinical data file")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--local_epochs", type=int, default=2,
                       help="Number of local epochs per round")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--num_classes", type=int, default=5,
                       help="Number of classes")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--mixed_precision", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to save best model")
    
    args = parser.parse_args()

    # Initialize trainer
    trainer = NVFlareTrainer(
        data_dir=args.data_dir,
        clinical_file=args.clinical_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        device=args.device,
        mixed_precision=args.mixed_precision,
        model_path=args.model_path,
    )

    # Initialize NVFlare client
    print("\n" + "="*60)
    print("Initializing NVFlare client...")
    print("="*60 + "\n")
    flare.init()

    client_id = flare.get_site_name()
    print(f"Client ID: {client_id}")

    # Federated learning loop
    print("\n" + "="*60)
    print("Starting federated learning...")
    print("="*60 + "\n")

    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        
        print(f"\n{'='*60}")
        print(f"Round {input_model.current_round}/{input_model.total_rounds}")
        print(f"{'='*60}")

        # Handle different tasks
        if flare.is_train():
            print(f"\n[{client_id}] Training task...")
            
            # Perform local training
            updated_weights = trainer.train_local(input_model.params)
            
            # Evaluate trained model
            train_metrics = trainer.evaluate_local(updated_weights)
            print(f"\n[{client_id}] Training complete:")
            print(f"  Local Val Loss: {train_metrics['loss']:.4f}")
            print(f"  Local Val Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Local Val F1: {train_metrics['f1']:.4f}")
            
            # Save best model
            trainer.save_best_model(updated_weights, train_metrics['accuracy'])
            
            # Send updated model back to server
            output_model = flare.FLModel(
                params=updated_weights,
                metrics=train_metrics,
                meta={
                    "client_id": client_id,
                    "num_samples": len(trainer.train_loader.dataset)
                }
            )
            flare.send(output_model)
            print(f"[{client_id}] Sent updated model to server")

        elif flare.is_evaluate():
            print(f"\n[{client_id}] Evaluation task...")
            
            # Evaluate received model
            eval_metrics = trainer.evaluate_local(input_model.params)
            print(f"\n[{client_id}] Evaluation complete:")
            print(f"  Val Loss: {eval_metrics['loss']:.4f}")
            print(f"  Val Acc: {eval_metrics['accuracy']:.4f}")
            print(f"  Val F1: {eval_metrics['f1']:.4f}")
            
            # Send evaluation metrics back to server
            output_model = flare.FLModel(
                params=input_model.params,  # Return same weights
                metrics=eval_metrics,
                meta={
                    "client_id": client_id,
                    "num_samples": len(trainer.val_loader.dataset)
                }
            )
            flare.send(output_model)
            print(f"[{client_id}] Sent evaluation metrics to server")

        elif flare.is_submit_model():
            print(f"\n[{client_id}] Submit model task...")
            
            # Load best local model
            if os.path.exists(trainer.model_path):
                best_weights = torch.load(trainer.model_path)
                print(f"[{client_id}] Loaded best local model from {trainer.model_path}")
            else:
                best_weights = trainer.model.state_dict()
                print(f"[{client_id}] No saved model found, using current weights")
            
            # Send best model
            output_model = flare.FLModel(
                params=best_weights,
                meta={
                    "client_id": client_id,
                    "best_accuracy": trainer.best_local_accuracy
                }
            )
            flare.send(output_model)
            print(f"[{client_id}] Sent best model to server")

        else:
            print(f"[{client_id}] Unknown task, skipping...")

    print(f"\n[{client_id}] Federated learning complete!")
    print(f"Best local accuracy: {trainer.best_local_accuracy:.4f}")


if __name__ == "__main__":
    main()
