import pytest
import yaml
from pathlib import Path

from oncolearn.config import (
    HuggingFaceConfig,
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
    cfg = ModalityConfig(name="tabular")
    assert cfg.name == "tabular"
    assert cfg.kwargs == {}


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.max_epochs == 50
    assert cfg.learning_rate == pytest.approx(1e-4)
    assert cfg.weight_decay == pytest.approx(1e-5)
    assert cfg.batch_size == 16
    assert cfg.num_workers == 4
    assert cfg.accelerator == "auto"
    assert cfg.devices == 1
    assert cfg.early_stopping_patience == 10
    assert cfg.subtype_lambda == pytest.approx(0.3)
    assert cfg.scheduler == "cosine"
    assert cfg.seed == 42


def test_output_config_defaults():
    cfg = OutputConfig()
    assert cfg.dir == "outputs"
    assert cfg.experiment_name == "experiment"
    assert cfg.save_every_n_epochs == 5


def test_huggingface_config_defaults():
    cfg = HuggingFaceConfig()
    assert cfg.model  # non-empty
    assert cfg.image_checkpoint is None


def test_oncolearn_config_optional_sections_get_defaults():
    cfg = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        modalities=[ModalityConfig(name="tabular")],
    )
    assert cfg.huggingface is None
    assert cfg.join_on == "patient_id"
    assert cfg.join_strategy == "inner"
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.output, OutputConfig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_passes_with_one_modality():
    cfg = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        modalities=[ModalityConfig(name="tabular")],
    )
    _validate(cfg)  # must not raise


def test_validate_passes_with_multiple_modalities():
    cfg = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        modalities=[ModalityConfig(name="tabular"), ModalityConfig(name="image")],
    )
    _validate(cfg)  # must not raise


def test_validate_raises_on_empty_modalities():
    cfg = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        modalities=[],
    )
    with pytest.raises(ValueError, match="at least one"):
        _validate(cfg)


def test_validate_raises_on_empty_model_name():
    cfg = OncoLearnConfig(
        model=ModelConfig(name=""),
        modalities=[ModalityConfig(name="tabular")],
    )
    with pytest.raises(ValueError, match="non-empty"):
        _validate(cfg)


def test_validate_raises_on_duplicate_modality_names():
    cfg = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        modalities=[ModalityConfig(name="tabular"), ModalityConfig(name="tabular")],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        _validate(cfg)


# ---------------------------------------------------------------------------
# load_config — happy paths
# ---------------------------------------------------------------------------

def test_load_minimal_config(tmp_path):
    """Only required sections — optional fields should get their defaults."""
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "gated_late_fusion"},
        "modalities": [{"name": "tabular"}],
    }))

    assert cfg.model.name == "gated_late_fusion"
    assert len(cfg.modalities) == 1
    assert cfg.modalities[0].name == "tabular"
    assert cfg.huggingface is None
    assert cfg.training.max_epochs == 50  # default


def test_load_full_config(tmp_path):
    """All sections populated — values are reflected exactly."""
    raw = {
        "model": {
            "name": "gated_late_fusion",
            "num_stage_classes": 3,
            "num_subtype_classes": 2,
            "freeze_encoders": False,
            "dropout": 0.1,
        },
        "modalities": [
            {
                "name": "tabular",
                "cohort_code": "TCGA-BRCA",
                "features_files": ["mirna.tsv", "pam50.tsv"],
            },
            {
                "name": "image",
                "n_slices": 7,
            },
        ],
        "training": {
            "max_epochs": 20,
            "learning_rate": 5e-4,
            "batch_size": 8,
            "seed": 123,
        },
        "huggingface": {
            "model": "some-org/some-model",
            "image_checkpoint": "/data/ckpt.pt",
        },
        "output": {
            "dir": "my_outputs",
            "experiment_name": "exp_01",
            "save_every_n_epochs": 10,
        },
        "join_on": "patient_id",
        "join_strategy": "inner",
    }
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", raw))

    assert cfg.model.num_stage_classes == 3
    assert cfg.model.num_subtype_classes == 2
    assert cfg.model.freeze_encoders is False
    assert cfg.model.dropout == pytest.approx(0.1)
    assert cfg.modalities[0].name == "tabular"
    assert cfg.modalities[0].kwargs["cohort_code"] == "TCGA-BRCA"
    assert cfg.modalities[0].kwargs["features_files"] == ["mirna.tsv", "pam50.tsv"]
    assert cfg.modalities[1].name == "image"
    assert cfg.modalities[1].kwargs["n_slices"] == 7
    assert cfg.training.max_epochs == 20
    assert cfg.training.seed == 123
    assert cfg.huggingface.model == "some-org/some-model"
    assert cfg.huggingface.image_checkpoint == "/data/ckpt.pt"
    assert cfg.output.dir == "my_outputs"
    assert cfg.output.experiment_name == "exp_01"
    assert cfg.output.save_every_n_epochs == 10


def test_load_config_without_huggingface_section(tmp_path):
    """huggingface is optional — omitting it yields hf=None."""
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "gated_late_fusion"},
        "modalities": [{"name": "tabular"}],
    }))
    assert cfg.huggingface is None


def test_load_config_with_huggingface_section(tmp_path):
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "modalities": [{"name": "tabular"}],
        "huggingface": {"model": "org/model"},
    }))
    assert cfg.huggingface is not None
    assert cfg.huggingface.model == "org/model"
    assert cfg.huggingface.image_checkpoint is None  # default


def test_load_config_partial_training_override(tmp_path):
    """Specified training fields are overridden; unspecified ones keep defaults."""
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "modalities": [{"name": "tabular"}],
        "training": {"max_epochs": 5, "seed": 99},
    }))
    assert cfg.training.max_epochs == 5
    assert cfg.training.seed == 99
    assert cfg.training.batch_size == 16   # default untouched
    assert cfg.training.scheduler == "cosine"  # default untouched


def test_load_config_modality_kwargs_are_inlined(tmp_path):
    """Modality kwargs sit at the same YAML level as 'name', not under a 'kwargs:' key."""
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "modalities": [{"name": "tabular", "cohort_code": "TCGA-BRCA", "n_workers": 2}],
    }))
    assert cfg.modalities[0].kwargs == {"cohort_code": "TCGA-BRCA", "n_workers": 2}


def test_load_config_unknown_training_keys_are_silently_dropped(tmp_path):
    """Unknown keys in any section must not cause errors."""
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "modalities": [{"name": "tabular"}],
        "training": {"max_epochs": 1, "nonexistent_key": "value"},
    }))
    assert cfg.training.max_epochs == 1


def test_load_config_join_fields(tmp_path):
    cfg = load_config(_write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "modalities": [{"name": "tabular"}],
        "join_on": "sample_id",
        "join_strategy": "inner",
    }))
    assert cfg.join_on == "sample_id"
    assert cfg.join_strategy == "inner"


# ---------------------------------------------------------------------------
# load_config — error paths
# ---------------------------------------------------------------------------

def test_load_config_raises_file_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config("/no/such/path/config.yaml")


def test_load_config_raises_on_missing_model_section(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "modalities": [{"name": "tabular"}],
    })
    with pytest.raises(KeyError, match="model"):
        load_config(cfg_file)


def test_load_config_raises_on_missing_modalities_section(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "gated_late_fusion"},
    })
    with pytest.raises(KeyError, match="modalities"):
        load_config(cfg_file)


def test_load_config_raises_on_empty_modalities_list(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "gated_late_fusion"},
        "modalities": [],
    })
    with pytest.raises(ValueError, match="at least one"):
        load_config(cfg_file)


def test_load_config_raises_on_duplicate_modality_names(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "modalities": [{"name": "tabular"}, {"name": "tabular"}],
    })
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(cfg_file)


def test_load_config_raises_on_modality_without_name(tmp_path):
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", {
        "model": {"name": "m"},
        "modalities": [{"cohort_code": "TCGA-BRCA"}],
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
    cfg = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        modalities=[ModalityConfig(name="tabular")],
    )
    out = tmp_path / "saved.yaml"
    save_config(cfg, out)
    assert out.exists()


def test_save_config_creates_parent_directories(tmp_path):
    cfg = OncoLearnConfig(
        model=ModelConfig(name="m"),
        modalities=[ModalityConfig(name="tabular")],
    )
    out = tmp_path / "nested" / "dir" / "cfg.yaml"
    save_config(cfg, out)
    assert out.exists()


def test_save_config_round_trips(tmp_path):
    """load → save → load should produce an equivalent config."""
    original = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion", num_stage_classes=3, dropout=0.1),
        modalities=[
            ModalityConfig(name="tabular", kwargs={"cohort_code": "TCGA-BRCA"}),
            ModalityConfig(name="image", kwargs={"n_slices": 7}),
        ],
        training=TrainingConfig(max_epochs=10, seed=7),
        huggingface=HuggingFaceConfig(model="org/model"),
        output=OutputConfig(dir="out", experiment_name="exp"),
    )
    path = tmp_path / "cfg.yaml"
    save_config(original, path)
    restored = load_config(path)

    assert restored.model.name == original.model.name
    assert restored.model.num_stage_classes == original.model.num_stage_classes
    assert restored.model.dropout == pytest.approx(original.model.dropout)
    assert len(restored.modalities) == len(original.modalities)
    assert restored.modalities[0].name == "tabular"
    assert restored.modalities[0].kwargs["cohort_code"] == "TCGA-BRCA"
    assert restored.modalities[1].name == "image"
    assert restored.modalities[1].kwargs["n_slices"] == 7
    assert restored.training.max_epochs == 10
    assert restored.training.seed == 7
    assert restored.huggingface.model == "org/model"
    assert restored.output.dir == "out"


def test_save_config_modality_kwargs_inlined_not_nested(tmp_path):
    """Saved YAML must store modality kwargs at the same level as 'name', not under 'kwargs:'."""
    cfg = OncoLearnConfig(
        model=ModelConfig(name="m"),
        modalities=[ModalityConfig(name="tabular", kwargs={"cohort_code": "TCGA-BRCA"})],
    )
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)

    with path.open() as f:
        raw = yaml.safe_load(f)

    assert raw["modalities"][0]["cohort_code"] == "TCGA-BRCA"
    assert "kwargs" not in raw["modalities"][0]


def test_save_config_omits_huggingface_when_none(tmp_path):
    """'huggingface' key must be absent from the YAML when huggingface=None."""
    cfg = OncoLearnConfig(
        model=ModelConfig(name="m"),
        modalities=[ModalityConfig(name="tabular")],
    )
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)

    with path.open() as f:
        raw = yaml.safe_load(f)

    assert "huggingface" not in raw


def test_save_config_includes_huggingface_when_set(tmp_path):
    cfg = OncoLearnConfig(
        model=ModelConfig(name="m"),
        modalities=[ModalityConfig(name="tabular")],
        huggingface=HuggingFaceConfig(model="org/model", image_checkpoint="/ckpt.pt"),
    )
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)

    with path.open() as f:
        raw = yaml.safe_load(f)

    assert raw["huggingface"]["model"] == "org/model"
    assert raw["huggingface"]["image_checkpoint"] == "/ckpt.pt"


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
    assert len(cfg.modalities) >= 1


def test_tabular_only_example_has_one_modality():
    cfg = load_config(DATA_CONFIGS / "tcga_brca_tabular_only.yaml")
    assert len(cfg.modalities) == 1
    assert cfg.modalities[0].name == "tabular"


def test_multimodal_example_has_tabular_and_image():
    cfg = load_config(DATA_CONFIGS / "tcga_brca_multimodal.yaml")
    names = {m.name for m in cfg.modalities}
    assert "tabular" in names
    assert "image" in names


def test_example_configs_have_huggingface_section():
    for filename in ("tcga_brca_tabular_only.yaml", "tcga_brca_multimodal.yaml"):
        cfg = load_config(DATA_CONFIGS / filename)
        assert cfg.huggingface is not None
        assert cfg.huggingface.model


def test_example_configs_have_valid_training_params():
    for filename in ("tcga_brca_tabular_only.yaml", "tcga_brca_multimodal.yaml"):
        cfg = load_config(DATA_CONFIGS / filename)
        assert cfg.training.max_epochs > 0
        assert cfg.training.learning_rate > 0
        assert cfg.training.batch_size > 0
        assert cfg.training.seed >= 0
