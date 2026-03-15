import pytest
import yaml
from pathlib import Path

from oncolearn.config import (
    DataConfig,
    EncoderConfig,
    ModelConfig,
    OncoLearnConfig,
    OutputConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from oncolearn.config.loader import _validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, content: dict) -> Path:
    with path.open("w") as f:
        yaml.dump(content, f)
    return path


def _make_config(**data_kwargs) -> OncoLearnConfig:
    """Build a minimal OncoLearnConfig using the new DataConfig structure."""
    pipeline = data_kwargs.pop("pipeline", "some_pipeline.py")
    return OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        data=DataConfig(pipeline=pipeline, **data_kwargs),
    )


REPO_ROOT = Path(__file__).parent.parent.parent
DATA_CONFIGS = REPO_ROOT / "data" / "configs" / "modeling" / "multimodal"


# ---------------------------------------------------------------------------
# Schema — default values
# ---------------------------------------------------------------------------

def test_model_config_defaults():
    cfg = ModelConfig(name="gated_late_fusion")
    assert cfg.num_stage_classes == 5
    assert cfg.num_subtype_classes == 0
    assert cfg.freeze_encoders is True
    assert cfg.dropout == pytest.approx(0.2)


def test_data_config_defaults():
    cfg = DataConfig(pipeline="my_pipeline.py")
    assert cfg.pipeline == "my_pipeline.py"
    assert cfg.splits_dir is None


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.max_epochs == 50
    assert cfg.optimizer.params["lr"] == pytest.approx(1e-4)
    assert cfg.optimizer.params["weight_decay"] == pytest.approx(1e-5)
    assert cfg.batch_size == 16
    assert cfg.num_workers == 4
    assert cfg.accelerator == "auto"
    assert cfg.devices == 1
    assert cfg.early_stopping_patience == 10
    assert cfg.subtype_lambda == pytest.approx(0.3)
    assert cfg.seed == 42


def test_output_config_defaults():
    cfg = OutputConfig()
    assert cfg.dir == "outputs"
    assert cfg.experiment_name == "experiment"
    assert cfg.save_every_n_epochs == 5


def test_oncolearn_config_optional_sections_get_defaults():
    cfg = _make_config()
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.output, OutputConfig)
    assert isinstance(cfg.data, DataConfig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_passes():
    cfg = _make_config()
    _validate(cfg)  # must not raise


def test_validate_raises_on_empty_pipeline():
    cfg = _make_config(pipeline="")
    with pytest.raises(ValueError, match="pipeline"):
        _validate(cfg)


def test_validate_raises_on_empty_model_name():
    cfg = OncoLearnConfig(
        model=ModelConfig(name=""),
        data=DataConfig(pipeline="some.py"),
    )
    with pytest.raises(ValueError, match="non-empty"):
        _validate(cfg)


# ---------------------------------------------------------------------------
# load_config — happy paths
# ---------------------------------------------------------------------------

def test_load_minimal_config(tmp_path):
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "gated_late_fusion"},
        "data": {"pipeline": "some_pipeline.py"},
    }))
    assert cfg.model.name == "gated_late_fusion"
    assert cfg.data.pipeline == "some_pipeline.py"
    assert cfg.training.max_epochs == 50  # default


def test_load_full_config(tmp_path):
    raw = {
        "model": {
            "name": "gated_late_fusion",
            "num_stage_classes": 4,
            "freeze_encoders": False,
            "dropout": 0.1,
            "encoders": [
                {"name": "gene", "modality": "oncolearn.modality.gene", "output_dim": 64},
            ],
        },
        "data": {
            "pipeline": "path/to/pipeline.py",
            "splits_dir": "data/splits/fold_0",
        },
        "training": {"max_epochs": 20, "batch_size": 8, "seed": 123},
        "output": {"dir": "my_outputs", "experiment_name": "exp_01"},
    }
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", raw))

    assert cfg.model.num_stage_classes == 4
    assert cfg.model.freeze_encoders is False
    assert cfg.model.encoders[0].modality == "oncolearn.modality.gene"
    assert cfg.data.pipeline == "path/to/pipeline.py"
    assert cfg.data.splits_dir == "data/splits/fold_0"
    assert cfg.training.max_epochs == 20
    assert cfg.training.seed == 123
    assert cfg.output.dir == "my_outputs"


def test_load_config_partial_training_override(tmp_path):
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "data": {"pipeline": "p.py"},
        "training": {"max_epochs": 5, "seed": 99},
    }))
    assert cfg.training.max_epochs == 5
    assert cfg.training.seed == 99
    assert cfg.training.batch_size == 16   # default untouched


def test_load_config_unknown_training_keys_are_silently_dropped(tmp_path):
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "data": {"pipeline": "p.py"},
        "training": {"max_epochs": 1, "nonexistent_key": "value"},
    }))
    assert cfg.training.max_epochs == 1


# ---------------------------------------------------------------------------
# load_config — error paths
# ---------------------------------------------------------------------------

def test_load_config_raises_file_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config("/no/such/path/config.yaml")


def test_load_config_raises_on_missing_model_section(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "data": {"pipeline": "p.py"},
    })
    with pytest.raises(KeyError, match="model"):
        load_config(cfg_file)


def test_load_config_raises_on_missing_data_section(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "gated_late_fusion"},
    })
    with pytest.raises(KeyError):
        load_config(cfg_file)


def test_load_config_raises_on_missing_pipeline(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "gated_late_fusion"},
        "data": {"splits_dir": None},
    })
    with pytest.raises((ValueError, KeyError)):
        load_config(cfg_file)


def test_load_config_raises_on_empty_file(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_config(empty)


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------

def test_save_config_creates_file(tmp_path):
    cfg = _make_config()
    out = tmp_path / "saved.yaml"
    save_config(cfg, out)
    assert out.exists()


def test_save_config_creates_parent_directories(tmp_path):
    cfg = _make_config()
    out = tmp_path / "nested" / "dir" / "cfg.yaml"
    save_config(cfg, out)
    assert out.exists()


def test_save_config_round_trips(tmp_path):
    original = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion", num_stage_classes=3, dropout=0.1),
        data=DataConfig(pipeline="path/to/pipeline.py", splits_dir="data/splits"),
        training=TrainingConfig(max_epochs=10, seed=7),
        output=OutputConfig(dir="out", experiment_name="exp"),
    )
    path = tmp_path / "cfg.yaml"
    save_config(original, path)
    restored = load_config(path)

    assert restored.model.name == original.model.name
    assert restored.model.num_stage_classes == original.model.num_stage_classes
    assert restored.model.dropout == pytest.approx(original.model.dropout)
    assert restored.data.pipeline == original.data.pipeline
    assert restored.data.splits_dir == original.data.splits_dir
    assert restored.training.max_epochs == 10
    assert restored.training.seed == 7
    assert restored.output.dir == "out"


def test_save_config_data_section_format(tmp_path):
    """Saved YAML has 'pipeline' key under 'data', not 'modalities'."""
    cfg = _make_config(pipeline="my/pipeline.py")
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)

    with path.open() as f:
        raw = yaml.safe_load(f)

    assert raw["data"]["pipeline"] == "my/pipeline.py"
    assert "modalities" not in raw["data"]


# ---------------------------------------------------------------------------
# Bundled example configs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    "tcga_brca_tabular_only.yaml",
    "tcga_brca_multimodal.yaml",
    "tcga_brca_cbioportal.yaml",
])
def test_example_configs_load_without_error(filename):
    cfg = load_config(DATA_CONFIGS / filename)
    assert cfg.model.name
    assert cfg.data.pipeline


def test_tabular_only_example_pipeline():
    cfg = load_config(DATA_CONFIGS / "tcga_brca_tabular_only.yaml")
    assert "tcga_brca_xenabrowser" in cfg.data.pipeline


def test_multimodal_example_has_gene_clinical_encoders():
    cfg = load_config(DATA_CONFIGS / "tcga_brca_multimodal.yaml")
    modalities = {e.modality for e in cfg.model.encoders}
    assert "oncolearn.modality.gene" in modalities
    assert "oncolearn.modality.clinical" in modalities


def test_example_configs_have_valid_training_params():
    for filename in ("tcga_brca_tabular_only.yaml", "tcga_brca_multimodal.yaml"):
        cfg = load_config(DATA_CONFIGS / filename)
        assert cfg.training.max_epochs > 0
        assert cfg.training.optimizer.params["lr"] > 0
        assert cfg.training.batch_size > 0
        assert cfg.training.seed >= 0
