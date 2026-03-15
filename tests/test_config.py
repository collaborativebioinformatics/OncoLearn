"""Tests for the pipeline-based config schema + loader (post-refactor)."""

import pytest
from pathlib import Path

from oncolearn.config.loader import load_config, save_config
from oncolearn.config.schema import (
    DataConfig,
    EncoderConfig,
    ModelConfig,
    OncoLearnConfig,
    TrainingConfig,
    OutputConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


MINIMAL_YAML = """\
model:
  name: gated_late_fusion
  num_stage_classes: 4
  encoders:
    - name: gene
      modality: oncolearn.modality.gene
      output_dim: 128

data:
  pipeline: data/configs/modeling/multimodal/preprocessing/tcga_brca_xenabrowser.py
"""

MULTI_ENCODER_YAML = """\
model:
  name: gated_late_fusion
  encoders:
    - name: gene
      modality: oncolearn.modality.gene
    - name: clinical
      modality: oncolearn.modality.clinical

data:
  pipeline: data/configs/modeling/multimodal/preprocessing/tcga_brca_xenabrowser.py
  splits_dir: data/splits/fold_0
"""


# ---------------------------------------------------------------------------
# DataConfig parsing
# ---------------------------------------------------------------------------

def test_load_data_section(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MINIMAL_YAML))
    assert isinstance(cfg.data, DataConfig)
    assert "tcga_brca_xenabrowser.py" in cfg.data.pipeline
    assert cfg.data.splits_dir is None


def test_load_splits_dir(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MULTI_ENCODER_YAML))
    assert cfg.data.splits_dir == "data/splits/fold_0"


def test_load_multiple_encoders(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MULTI_ENCODER_YAML))
    assert len(cfg.model.encoders) == 2
    modalities = [e.modality for e in cfg.model.encoders]
    assert "oncolearn.modality.gene" in modalities
    assert "oncolearn.modality.clinical" in modalities


# ---------------------------------------------------------------------------
# EncoderConfig.modality
# ---------------------------------------------------------------------------

def test_encoder_modality_populated(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MINIMAL_YAML))
    enc = cfg.model.encoders[0]
    assert enc.modality == "oncolearn.modality.gene"


def test_encoder_modality_absent(tmp_path):
    yaml_str = """\
model:
  name: gated_late_fusion
  encoders:
    - name: gene
      output_dim: 64

data:
  pipeline: some_pipeline.py
"""
    cfg = load_config(write_yaml(tmp_path, yaml_str))
    assert cfg.model.encoders[0].modality is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validation_empty_pipeline(tmp_path):
    yaml_str = """\
model:
  name: gated_late_fusion

data:
  pipeline: ""
"""
    with pytest.raises((ValueError, KeyError)):
        load_config(write_yaml(tmp_path, yaml_str))


def test_validation_missing_pipeline(tmp_path):
    yaml_str = """\
model:
  name: gated_late_fusion

data:
  splits_dir: null
"""
    with pytest.raises((ValueError, KeyError)):
        load_config(write_yaml(tmp_path, yaml_str))


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MINIMAL_YAML))
    out_path = tmp_path / "out.yaml"
    save_config(cfg, out_path)
    cfg2 = load_config(out_path)

    assert cfg2.data.pipeline == cfg.data.pipeline
    assert cfg2.data.splits_dir == cfg.data.splits_dir
    assert cfg2.model.encoders[0].modality == cfg.model.encoders[0].modality
