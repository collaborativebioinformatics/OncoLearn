"""
Tests for cross-validation and HPO training behavior.

All tests use unittest.mock — no real data, models, or Optuna studies required.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from oncolearn.config.schema import (
    CrossValidationConfig,
    DataConfig,
    HpoConfig,
    ModelConfig,
    OncoLearnConfig,
    OutputConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(cv_enabled=False, folds_dirs=None, hpo=None):
    """Minimal OncoLearnConfig for testing — no real files required."""
    return OncoLearnConfig(
        model=ModelConfig(name="fake_model"),
        data=DataConfig(pipeline="fake/pipeline.py"),
        training=TrainingConfig(
            max_epochs=1,
            accelerator="cpu",
            cross_validation=CrossValidationConfig(
                enabled=cv_enabled,
                folds_dirs=folds_dirs or [],
            ),
            hpo=hpo,
        ),
        output=OutputConfig(dir="/tmp/test_cv_hpo", experiment_name="test_exp"),
    )


# ---------------------------------------------------------------------------
# OncoTrainer CV vs non-CV
# ---------------------------------------------------------------------------


def test_cv_trainer_skips_validation():
    """CV mode: PL Trainer constructed with val suppressed; .test() called after fit."""
    from oncolearn.trainer import OncoTrainer

    config = _make_config(cv_enabled=True)

    mock_pl = MagicMock()
    mock_pl.callback_metrics = {}
    mock_pl.test.return_value = [{"test_acc": 0.9}]

    with patch.object(OncoTrainer, "_build_datamodule", return_value=MagicMock()), \
         patch.object(OncoTrainer, "_build_model", return_value=MagicMock()), \
         patch("oncolearn.trainer.TensorBoardLogger"), \
         patch("pytorch_lightning.Trainer", return_value=mock_pl) as MockTrainer:

        trainer = OncoTrainer(config)
        metrics = trainer.train()

    kw = MockTrainer.call_args[1]
    assert kw["limit_val_batches"] == 0.0
    assert kw["num_sanity_val_steps"] == 0
    mock_pl.fit.assert_called_once()
    mock_pl.test.assert_called_once()
    assert "test_acc" in metrics


def test_non_cv_trainer_uses_val():
    """Non-CV mode: limit_val_batches=1.0, EarlyStopping present, .test() not called."""
    from oncolearn.trainer import OncoTrainer
    from pytorch_lightning.callbacks import EarlyStopping

    config = _make_config(cv_enabled=False)

    mock_pl = MagicMock()
    mock_pl.callback_metrics = {}

    with patch.object(OncoTrainer, "_build_datamodule", return_value=MagicMock()), \
         patch.object(OncoTrainer, "_build_model", return_value=MagicMock()), \
         patch("oncolearn.trainer.TensorBoardLogger"), \
         patch("pytorch_lightning.Trainer", return_value=mock_pl) as MockTrainer:

        trainer = OncoTrainer(config)
        trainer.train()

    kw = MockTrainer.call_args[1]
    assert kw["limit_val_batches"] == 1.0
    callbacks = kw["callbacks"]
    assert any(isinstance(cb, EarlyStopping) for cb in callbacks), \
        "EarlyStopping must be in callbacks for non-CV mode"
    mock_pl.test.assert_not_called()


# ---------------------------------------------------------------------------
# _TrialOncoTrainer CV mode vs non-CV mode
# ---------------------------------------------------------------------------


def test_trial_trainer_cv_mode_skips_val():
    """_TrialOncoTrainer(cv_mode=True): val suppressed, .test() called, metric returned."""
    from oncolearn.modeling.hyps.tuner import _TrialOncoTrainer

    config = _make_config()

    mock_pl = MagicMock()
    mock_pl.callback_metrics = {}
    mock_pl.test.return_value = [{"test_f1": 0.75}]

    with patch("oncolearn.trainer.OncoTrainer.__init__", return_value=None), \
         patch("pytorch_lightning.Trainer", return_value=mock_pl) as MockTrainer:

        trial_trainer = _TrialOncoTrainer(config, cv_mode=True)
        # Replace the empty OncoTrainer stub with a proper mock
        trial_trainer._trainer = MagicMock()
        metrics = trial_trainer.train()

    kw = MockTrainer.call_args[1]
    assert kw["limit_val_batches"] == 0.0
    assert kw["num_sanity_val_steps"] == 0
    mock_pl.fit.assert_called_once()
    mock_pl.test.assert_called_once()
    assert metrics.get("test_f1") == pytest.approx(0.75)


def test_trial_trainer_non_cv_mode_uses_val():
    """_TrialOncoTrainer(cv_mode=False): EarlyStopping present, .test() not called."""
    from oncolearn.modeling.hyps.tuner import _TrialOncoTrainer
    from pytorch_lightning.callbacks import EarlyStopping

    config = _make_config()

    mock_pl = MagicMock()
    mock_pl.callback_metrics = {}

    with patch("oncolearn.trainer.OncoTrainer.__init__", return_value=None), \
         patch("pytorch_lightning.Trainer", return_value=mock_pl) as MockTrainer:

        trial_trainer = _TrialOncoTrainer(config, cv_mode=False)
        trial_trainer._trainer = MagicMock()
        trial_trainer.train()

    kw = MockTrainer.call_args[1]
    assert kw["limit_val_batches"] == 1.0
    callbacks = kw["callbacks"]
    assert any(isinstance(cb, EarlyStopping) for cb in callbacks), \
        "EarlyStopping must be in callbacks for non-CV _TrialOncoTrainer"
    mock_pl.test.assert_not_called()


# ---------------------------------------------------------------------------
# OptunaHPTuner CV fold averaging
# ---------------------------------------------------------------------------


def test_hpo_cv_averages_test_metric():
    """HPO + CV: _objective() returns the mean test metric across folds."""
    from oncolearn.modeling.hyps.tuner import OptunaHPTuner

    hpo_cfg = HpoConfig(metric="test_f1", n_trials=1, pruning=False)
    config = _make_config(
        cv_enabled=True,
        folds_dirs=["fold_0", "fold_1"],
        hpo=hpo_cfg,
    )

    tuner = OptunaHPTuner(config, hpo_cfg)

    mock_trial = MagicMock()
    mock_trial.number = 0
    mock_trial.params = {}

    mock_trial_instance = MagicMock()
    mock_trial_instance.train.side_effect = [{"test_f1": 0.8}, {"test_f1": 0.6}]

    with patch("oncolearn.modeling.hyps.tuner._TrialOncoTrainer", return_value=mock_trial_instance), \
         patch("oncolearn.modeling.hyps.search_space.suggest_hyperparams", return_value=config):
        result = tuner._objective(mock_trial)

    assert result == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Schema regression guard
# ---------------------------------------------------------------------------


def test_cv_config_no_metric_field():
    """CrossValidationConfig must not have a 'metric' field — that belongs in HpoConfig."""
    cv = CrossValidationConfig(enabled=True)
    assert not hasattr(cv, "metric"), \
        "CrossValidationConfig must not have 'metric' — it belongs in HpoConfig"

    h = HpoConfig()
    assert h.metric == "val_acc"
