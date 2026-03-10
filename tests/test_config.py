"""Tests for schema + loader restructuring (Unit 2)."""

import pytest
from pathlib import Path

from oncolearn.config.loader import load_config, save_config
from oncolearn.config.schema import (
    DataConfig,
    EncoderConfig,
    ModalityConfig,
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
  num_stage_classes: 5
  encoders:
    - name: gene
      modality: oncolearn.modality.gene
      output_dim: 128

data:
  base_directory: data/xenabrowser
  cohort_code: TCGA-BRCA
  modalities:
    - name: oncolearn.modality.gene
      join_on: patient_id
      join_strategy: inner
      files:
        - TCGA-BRCA.mirna.tsv
        - pam50.tsv
"""

MULTI_MODALITY_YAML = """\
model:
  name: gated_late_fusion
  encoders:
    - name: gene
      modality: oncolearn.modality.gene
    - name: clinical
      modality: oncolearn.modality.clinical

data:
  base_directory: data/xena
  cohort_code: TCGA-BRCA
  splits_dir: data/splits/fold_0
  modalities:
    - name: oncolearn.modality.gene
      join_on: patient_id
      join_strategy: inner
      files:
        - a.tsv
        - b.tsv
    - name: oncolearn.modality.clinical
      files:
        - c.tsv
"""


# ---------------------------------------------------------------------------
# DataConfig parsing
# ---------------------------------------------------------------------------

def test_load_data_section(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MINIMAL_YAML))

    assert isinstance(cfg.data, DataConfig)
    assert cfg.data.base_directory == "data/xenabrowser"
    assert cfg.data.cohort_code == "TCGA-BRCA"
    assert cfg.data.splits_dir is None
    assert len(cfg.data.modalities) == 1


def test_load_modality_name(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MINIMAL_YAML))
    mod = cfg.data.modalities[0]
    assert mod.name == "oncolearn.modality.gene"


def test_load_modality_files(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MINIMAL_YAML))
    mod = cfg.data.modalities[0]
    assert mod.files == ["TCGA-BRCA.mirna.tsv", "pam50.tsv"]


def test_load_modality_join_on_default(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MINIMAL_YAML))
    assert cfg.data.modalities[0].join_on == "patient_id"


def test_load_modality_join_on_custom(tmp_path):
    yaml_str = MINIMAL_YAML.replace("join_on: patient_id", "join_on: sample_id")
    cfg = load_config(write_yaml(tmp_path, yaml_str))
    assert cfg.data.modalities[0].join_on == "sample_id"


def test_load_splits_dir(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MULTI_MODALITY_YAML))
    assert cfg.data.splits_dir == "data/splits/fold_0"


def test_load_multiple_modalities(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MULTI_MODALITY_YAML))
    assert len(cfg.data.modalities) == 2
    names = [m.name for m in cfg.data.modalities]
    assert "oncolearn.modality.gene" in names
    assert "oncolearn.modality.clinical" in names


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
  modalities:
    - name: oncolearn.modality.gene
"""
    cfg = load_config(write_yaml(tmp_path, yaml_str))
    assert cfg.model.encoders[0].modality is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validation_encoder_modality_mismatch(tmp_path):
    yaml_str = """\
model:
  name: gated_late_fusion
  encoders:
    - name: gene
      modality: oncolearn.modality.NONEXISTENT

data:
  modalities:
    - name: oncolearn.modality.gene
"""
    with pytest.raises(ValueError, match="references modality"):
        load_config(write_yaml(tmp_path, yaml_str))


def test_validation_empty_modalities(tmp_path):
    yaml_str = """\
model:
  name: gated_late_fusion

data:
  modalities: []
"""
    with pytest.raises((ValueError, KeyError)):
        load_config(write_yaml(tmp_path, yaml_str))


def test_validation_duplicate_modality_names(tmp_path):
    yaml_str = """\
model:
  name: gated_late_fusion

data:
  modalities:
    - name: oncolearn.modality.gene
    - name: oncolearn.modality.gene
"""
    with pytest.raises(ValueError, match="Duplicate modality names"):
        load_config(write_yaml(tmp_path, yaml_str))


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    cfg = load_config(write_yaml(tmp_path, MINIMAL_YAML))
    out_path = tmp_path / "out.yaml"
    save_config(cfg, out_path)
    cfg2 = load_config(out_path)

    assert cfg2.data.base_directory == cfg.data.base_directory
    assert cfg2.data.cohort_code == cfg.data.cohort_code
    assert len(cfg2.data.modalities) == len(cfg.data.modalities)
    assert cfg2.data.modalities[0].name == cfg.data.modalities[0].name
    assert cfg2.data.modalities[0].files == cfg.data.modalities[0].files
    assert cfg2.model.encoders[0].modality == cfg.model.encoders[0].modality
