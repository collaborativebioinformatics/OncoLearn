"""
Utility functions for training and evaluation.
"""
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load YAML config with optional overrides and base config support."""
    config_dir = Path(config_path).parent
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle _base_ inheritance
    if '_base_' in config:
        base_path = config_dir / config['_base_']
        base_config = load_config(str(base_path))
        # Merge: base first, then current config
        merged = {**base_config, **config}
        # Remove _base_ key
        merged.pop('_base_', None)
        config = merged
    
    if overrides:
        config.update(overrides)
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """Save config to YAML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logging(log_dir: Path, variant: str, fold: int):
    """Setup logging to file and stdout."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{variant}_fold_{fold}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_jsonl(data: Dict, filepath: str):
    """Append data to JSONL file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(json.dumps(data) + '\n')

