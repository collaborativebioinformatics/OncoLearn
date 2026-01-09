"""
Configuration management utilities.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    """
    Configuration for cancer subtyping model training.
    """

    # Data parameters
    data_dir: str = "data/TCIA"
    clinical_file: Optional[str] = "data/GDCdata/TCGA-BRCA.clinical.tsv"
    genetic_data_dir: Optional[str] = "/workspace/data/processed"
    image_size: tuple = (512, 512)
    batch_size: int = 16
    num_workers: int = 4
    cancer_type: Optional[str] = None  # e.g., "BRCA", "LUAD", etc.
    label_column: str = "ajcc_pathologic_stage.diagnoses"
    use_genetic_data: bool = True
    max_genes: Optional[int] = 1000
    image_extension: str = "*.png"

    # Model parameters
    yolo_model_name: str = "hustvl/yolos-tiny"
    yolo_pretrained: bool = True
    freeze_yolo: bool = False
    num_attention_layers: int = 2
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    latent_dim: int = 128
    vae_hidden_dims: list = field(default_factory=lambda: [512, 256])
    vae_dropout: float = 0.1
    num_classes: int = 5
    classifier_hidden_dim: Optional[int] = None
    classifier_dropout: float = 0.1
    use_vae_loss: bool = True
    kld_weight: float = 0.1

    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau'
    scheduler_patience: int = 10
    early_stopping_patience: int = 20
    gradient_clip_val: float = 1.0
    classification_weight: float = 1.0

    # Logging and checkpointing
    output_dir: str = "outputs"
    experiment_name: str = "cancer_subtyping"
    log_interval: int = 10
    save_interval: int = 5
    save_best_only: bool = True

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True

    # NVIDIA FLARE parameters
    use_nvflare: bool = False
    nvflare_workspace: Optional[str] = None
    client_id: Optional[str] = None

    # Random seed
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save config to file (supports .yaml or .json)."""
        path = Path(path)
        config_dict = self.to_dict()

        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        # Handle tuple conversion
        if 'image_size' in config_dict and isinstance(config_dict['image_size'], list):
            config_dict['image_size'] = tuple(config_dict['image_size'])
        return cls(**config_dict)

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from file."""
        path = Path(path)

        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return cls.from_dict(config_dict)


def load_config(path: str) -> Config:
    """
    Load configuration from file.

    Args:
        path: Path to config file (.yaml or .json)

    Returns:
        Config object
    """
    return Config.load(path)


def save_config(config: Config, path: str):
    """
    Save configuration to file.

    Args:
        config: Config object
        path: Path to save config (.yaml or .json)
    """
    config.save(path)


def create_default_config(output_path: Optional[str] = None) -> Config:
    """
    Create default configuration.

    Args:
        output_path: Optional path to save config

    Returns:
        Default Config object
    """
    config = Config()

    if output_path:
        config.save(output_path)

    return config
