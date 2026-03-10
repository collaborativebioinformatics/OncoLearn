import pytest
import yaml
from pathlib import Path

from oncolearn.config import (
    DataConfig,
    ModalityConfig,
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
    modalities = data_kwargs.pop("modalities", [ModalityConfig(name="gene")])
    return OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        data=DataConfig(modalities=modalities, **data_kwargs),
    )


REPO_ROOT = Path(__file__).parent.parent.parent
DATA_CONFIGS = REPO_ROOT / "data" / "configs"


# ---------------------------------------------------------------------------
# Schema — default values
# ---------------------------------------------------------------------------

def test_model_config_defaults():
    cfg = ModelConfig(name="gated_late_fusion")
    assert cfg.num_stage_classes == 5
    assert cfg.num_subtype_classes == 0
    assert cfg.freeze_encoders is True
    assert cfg.dropout == pytest.approx(0.2)


def test_modality_config_defaults():
    cfg = ModalityConfig(name="oncolearn.modality.gene")
    assert cfg.name == "oncolearn.modality.gene"
    assert cfg.join_on == "patient_id"
    assert cfg.join_strategy == "inner"
    assert cfg.files is None
    assert cfg.kwargs == {}


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


def test_data_config_defaults():
    cfg = DataConfig(modalities=[ModalityConfig(name="gene")])
    assert cfg.base_directory == "data/xenabrowser"
    assert cfg.cohort_code == "TCGA-BRCA"
    assert cfg.splits_dir is None


def test_oncolearn_config_optional_sections_get_defaults():
    cfg = _make_config()
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.output, OutputConfig)
    assert isinstance(cfg.data, DataConfig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_passes_with_one_modality():
    cfg = _make_config()
    _validate(cfg)  # must not raise


def test_validate_passes_with_multiple_modalities():
    cfg = _make_config(modalities=[
        ModalityConfig(name="gene"),
        ModalityConfig(name="image"),
    ])
    _validate(cfg)  # must not raise


def test_validate_raises_on_empty_modalities():
    cfg = _make_config(modalities=[])
    with pytest.raises(ValueError, match="at least one"):
        _validate(cfg)


def test_validate_raises_on_empty_model_name():
    cfg = OncoLearnConfig(
        model=ModelConfig(name=""),
        data=DataConfig(modalities=[ModalityConfig(name="gene")]),
    )
    with pytest.raises(ValueError, match="non-empty"):
        _validate(cfg)


def test_validate_raises_on_duplicate_modality_names():
    cfg = _make_config(modalities=[
        ModalityConfig(name="gene"),
        ModalityConfig(name="gene"),
    ])
    with pytest.raises(ValueError, match="Duplicate"):
        _validate(cfg)


def test_validate_raises_when_encoder_modality_not_in_data(tmp_path):
    """encoder.modality must match a name in data.modalities."""
    from oncolearn.config.schema import EncoderConfig
    cfg = OncoLearnConfig(
        model=ModelConfig(
            name="gated_late_fusion",
            encoders=[EncoderConfig(name="gene", modality="oncolearn.modality.NONEXISTENT")],
        ),
        data=DataConfig(modalities=[ModalityConfig(name="oncolearn.modality.gene")]),
    )
    with pytest.raises(ValueError, match="references modality"):
        _validate(cfg)


# ---------------------------------------------------------------------------
# load_config — happy paths (new data: section format)
# ---------------------------------------------------------------------------

def test_load_minimal_config_new_format(tmp_path):
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "gated_late_fusion"},
        "data": {
            "modalities": [{"name": "oncolearn.modality.gene"}],
        },
    }))
    assert cfg.model.name == "gated_late_fusion"
    assert len(cfg.data.modalities) == 1
    assert cfg.data.modalities[0].name == "oncolearn.modality.gene"
    assert cfg.training.max_epochs == 50  # default


def test_load_full_config_new_format(tmp_path):
    raw = {
        "model": {
            "name": "gated_late_fusion",
            "num_stage_classes": 3,
            "num_subtype_classes": 2,
            "freeze_encoders": False,
            "dropout": 0.1,
            "encoders": [
                {"name": "gene", "modality": "oncolearn.modality.gene", "output_dim": 64},
            ],
        },
        "data": {
            "base_directory": "data/xena",
            "cohort_code": "TCGA-BRCA",
            "splits_dir": "data/splits/fold_0",
            "modalities": [
                {
                    "name": "oncolearn.modality.gene",
                    "join_on": "patient_id",
                    "join_strategy": "inner",
                    "files": ["mirna.tsv", "pam50.tsv"],
                },
                {
                    "name": "oncolearn.modality.image",
                    "n_slices": 7,
                },
            ],
        },
        "training": {"max_epochs": 20, "batch_size": 8, "seed": 123},
        "output": {"dir": "my_outputs", "experiment_name": "exp_01"},
    }
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", raw))

    assert cfg.model.num_stage_classes == 3
    assert cfg.model.freeze_encoders is False
    assert cfg.model.encoders[0].modality == "oncolearn.modality.gene"
    assert cfg.data.base_directory == "data/xena"
    assert cfg.data.splits_dir == "data/splits/fold_0"
    assert cfg.data.modalities[0].name == "oncolearn.modality.gene"
    assert cfg.data.modalities[0].files == ["mirna.tsv", "pam50.tsv"]
    assert cfg.data.modalities[1].kwargs["n_slices"] == 7
    assert cfg.training.max_epochs == 20
    assert cfg.training.seed == 123
    assert cfg.output.dir == "my_outputs"


def test_load_config_partial_training_override(tmp_path):
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "data": {"modalities": [{"name": "gene"}]},
        "training": {"max_epochs": 5, "seed": 99},
    }))
    assert cfg.training.max_epochs == 5
    assert cfg.training.seed == 99
    assert cfg.training.batch_size == 16   # default untouched


def test_load_config_unknown_training_keys_are_silently_dropped(tmp_path):
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "data": {"modalities": [{"name": "gene"}]},
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
        "data": {"modalities": [{"name": "gene"}]},
    })
    with pytest.raises(KeyError, match="model"):
        load_config(cfg_file)


def test_load_config_raises_on_missing_data_section(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "gated_late_fusion"},
    })
    with pytest.raises(KeyError):
        load_config(cfg_file)


def test_load_config_raises_on_empty_modalities_list(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "gated_late_fusion"},
        "data": {"modalities": []},
    })
    with pytest.raises((ValueError, KeyError)):
        load_config(cfg_file)


def test_load_config_raises_on_duplicate_modality_names(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "data": {"modalities": [{"name": "gene"}, {"name": "gene"}]},
    })
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(cfg_file)


def test_load_config_raises_on_modality_without_name(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "data": {"modalities": [{"cohort_code": "TCGA-BRCA"}]},
    })
    with pytest.raises(KeyError, match="name"):
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
        data=DataConfig(
            modalities=[
                ModalityConfig(name="gene", files=["a.tsv"], kwargs={"cohort_code": "TCGA-BRCA"}),
                ModalityConfig(name="image", kwargs={"n_slices": 7}),
            ],
        ),
        training=TrainingConfig(max_epochs=10, seed=7),
        output=OutputConfig(dir="out", experiment_name="exp"),
    )
    path = tmp_path / "cfg.yaml"
    save_config(original, path)
    restored = load_config(path)

    assert restored.model.name == original.model.name
    assert restored.model.num_stage_classes == original.model.num_stage_classes
    assert restored.model.dropout == pytest.approx(original.model.dropout)
    assert len(restored.data.modalities) == len(original.data.modalities)
    assert restored.data.modalities[0].name == "gene"
    assert restored.data.modalities[0].files == ["a.tsv"]
    assert restored.data.modalities[0].kwargs["cohort_code"] == "TCGA-BRCA"
    assert restored.data.modalities[1].name == "image"
    assert restored.data.modalities[1].kwargs["n_slices"] == 7
    assert restored.training.max_epochs == 10
    assert restored.training.seed == 7
    assert restored.output.dir == "out"


def test_save_config_modality_kwargs_inlined_not_nested(tmp_path):
    """Saved YAML inlines modality kwargs alongside 'name' under data.modalities."""
    cfg = _make_config(modalities=[
        ModalityConfig(name="gene", kwargs={"cohort_code": "TCGA-BRCA"}),
    ])
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)

    with path.open() as f:
        raw = yaml.safe_load(f)

    assert raw["data"]["modalities"][0]["cohort_code"] == "TCGA-BRCA"
    assert "kwargs" not in raw["data"]["modalities"][0]


# ---------------------------------------------------------------------------
# Bundled example configs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    "tcga_brca_tabular_only.yaml",
    "tcga_brca_multimodal.yaml",
])
def test_example_configs_load_without_error(filename):
    cfg = load_config(DATA_CONFIGS / filename)
    assert cfg.model.name
    assert len(cfg.data.modalities) >= 1


def test_tabular_only_example_has_one_modality():
    cfg = load_config(DATA_CONFIGS / "tcga_brca_tabular_only.yaml")
    assert len(cfg.data.modalities) == 1
    assert cfg.data.modalities[0].name == "oncolearn.modality.gene"


def test_multimodal_example_has_gene_clinical_image():
    cfg = load_config(DATA_CONFIGS / "tcga_brca_multimodal.yaml")
    names = {m.name for m in cfg.data.modalities}
    assert "oncolearn.modality.gene" in names
    assert "oncolearn.modality.clinical" in names
    assert "oncolearn.modality.image" in names


def test_example_configs_have_valid_training_params():
    for filename in ("tcga_brca_tabular_only.yaml", "tcga_brca_multimodal.yaml"):
        cfg = load_config(DATA_CONFIGS / filename)
        assert cfg.training.max_epochs > 0
        assert cfg.training.optimizer.params["lr"] > 0
        assert cfg.training.batch_size > 0
        assert cfg.training.seed >= 0
