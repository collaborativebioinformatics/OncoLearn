"""
Optuna-based hyperparameter tuner for OncoLearn.

Called automatically by :class:`~oncolearn.trainer.OncoTrainer` when
``training.hpo`` is present in the experiment config.  Not normally
instantiated directly.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import optuna
import pytorch_lightning as pl

if TYPE_CHECKING:
    from oncolearn.config import HpoConfig, OncoLearnConfig

logger = logging.getLogger(__name__)


class OptunaHPTuner:
    """Run an Optuna study driven entirely by :class:`~oncolearn.config.HpoConfig`.

    Args:
        base_config: Structural config template (data paths, encoder names, etc.).
        hpo_cfg:     Parsed HPO settings from ``training.hpo``.
    """

    def __init__(
        self,
        base_config: "OncoLearnConfig",
        hpo_cfg: "HpoConfig",
    ) -> None:
        self.base_config = base_config
        self.hpo_cfg = hpo_cfg
        self._study: Optional[optuna.Study] = None

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective(self, trial: optuna.Trial) -> float:
        from oncolearn.trainer import OncoTrainer, set_seed
        from .search_space import suggest_hyperparams

        config = suggest_hyperparams(trial, self.base_config, self.hpo_cfg)

        if self.hpo_cfg.epochs_per_trial is not None:
            config.training.max_epochs = self.hpo_cfg.epochs_per_trial

        set_seed(self.hpo_cfg.seed + trial.number)

        n = self.hpo_cfg.n_trials
        print(
            f"\n[HPO] Trial {trial.number + 1}/{n} | params={trial.params}",
            flush=True,
        )

        cv = self.base_config.training.cross_validation
        if cv.enabled and cv.folds_dirs:
            fold_values = []
            for fold_idx, fold_dir in enumerate(cv.folds_dirs):
                fold_config = copy.deepcopy(config)
                fold_config.data.splits_dir = fold_dir
                fold_config.output.experiment_name = (
                    f"{self.hpo_cfg.study_name}/trial_{trial.number}/fold_{fold_idx}"
                )
                try:
                    fold_trainer = _TrialOncoTrainer(fold_config, cv_mode=True)
                    fold_metrics = fold_trainer.train()
                except Exception as exc:
                    logger.warning(
                        "Trial %d fold %d failed: %s", trial.number, fold_idx, exc
                    )
                    raise optuna.exceptions.TrialPruned() from exc
                v = fold_metrics.get(self.hpo_cfg.metric)
                if v is None:
                    logger.warning(
                        "Trial %d fold %d: metric '%s' not found. Available: %s",
                        trial.number,
                        fold_idx,
                        self.hpo_cfg.metric,
                        list(fold_metrics.keys()),
                    )
                    raise optuna.exceptions.TrialPruned()
                fold_values.append(float(v.item()) if hasattr(v, "item") else float(v))
            value = sum(fold_values) / len(fold_values)
            print(
                f"[HPO] Trial {trial.number + 1} CV | "
                f"mean {self.hpo_cfg.metric}={value:.4f} | per-fold={fold_values}",
                flush=True,
            )
            logger.info(
                "Trial %d CV | mean %s=%.4f | per-fold=%s | params=%s",
                trial.number,
                self.hpo_cfg.metric,
                value,
                fold_values,
                trial.params,
            )
            return value

        config.output.experiment_name = (
            f"{self.hpo_cfg.study_name}/trial_{trial.number}"
        )

        extra_callbacks: List[pl.Callback] = []
        if self.hpo_cfg.pruning:
            cb = _make_pruning_callback(trial, self.hpo_cfg.metric)
            if cb is not None:
                extra_callbacks.append(cb)

        try:
            trainer = _TrialOncoTrainer(config, extra_callbacks=extra_callbacks)
            metrics = trainer.train()
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            logger.warning("Trial %d failed: %s", trial.number, exc, exc_info=True)
            raise optuna.exceptions.TrialPruned() from exc

        value = metrics.get(self.hpo_cfg.metric)
        if value is None:
            logger.warning(
                "Trial %d: metric '%s' not found. Available: %s",
                trial.number,
                self.hpo_cfg.metric,
                list(metrics.keys()),
            )
            raise optuna.exceptions.TrialPruned()

        if hasattr(value, "item"):
            value = value.item()

        print(
            f"[HPO] Trial {trial.number + 1}/{self.hpo_cfg.n_trials} done | "
            f"{self.hpo_cfg.metric}={value:.4f}",
            flush=True,
        )
        logger.info(
            "Trial %d | %s=%.4f | params=%s",
            trial.number,
            self.hpo_cfg.metric,
            value,
            trial.params,
        )
        return float(value)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tune(self) -> Tuple[Dict, "OncoLearnConfig"]:
        """Run the study and return ``(best_params, best_config)``.

        ``best_config`` has the best-found values already applied and is
        ready to pass straight to :class:`~oncolearn.trainer.OncoTrainer`
        for a final full training run.
        """
        from .search_space import apply_params

        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
            if self.hpo_cfg.pruning
            else optuna.pruners.NopPruner()
        )
        sampler = optuna.samplers.TPESampler(seed=self.hpo_cfg.seed)

        self._study = optuna.create_study(
            study_name=self.hpo_cfg.study_name,
            storage=self.hpo_cfg.storage,
            direction=self.hpo_cfg.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        optuna.logging.set_verbosity(
            optuna.logging.DEBUG
            if logger.isEnabledFor(logging.DEBUG)
            else optuna.logging.WARNING
        )

        logger.info(
            "HPO start | study=%s | n_trials=%d | metric=%s | direction=%s",
            self.hpo_cfg.study_name,
            self.hpo_cfg.n_trials,
            self.hpo_cfg.metric,
            self.hpo_cfg.direction,
        )

        self._study.optimize(
            self._objective,
            n_trials=self.hpo_cfg.n_trials,
            catch=(Exception,),
        )

        best_trial = self._study.best_trial

        from .search_space import config_params_from_trial
        best_params = config_params_from_trial(best_trial.params, self.hpo_cfg)

        best_config = copy.deepcopy(self.base_config)
        apply_params(best_config, best_params)

        logger.info(
            "HPO done | best %s=%.4f | params=%s",
            self.hpo_cfg.metric,
            best_trial.value,
            best_params,
        )
        return best_params, best_config

    @property
    def study(self) -> Optional[optuna.Study]:
        """Underlying :class:`optuna.Study` (available after :meth:`tune`)."""
        return self._study

    def results_dataframe(self):
        """Return a pandas DataFrame of all completed trials."""
        if self._study is None:
            raise RuntimeError("Call tune() first.")
        return self._study.trials_dataframe()


# ---------------------------------------------------------------------------
# Internal: trial trainer with injectable callbacks
# ---------------------------------------------------------------------------

class _TrialOncoTrainer:
    """Thin wrapper around OncoTrainer that injects extra PL callbacks."""

    def __init__(
        self,
        config: "OncoLearnConfig",
        extra_callbacks: Optional[List[pl.Callback]] = None,
        cv_mode: bool = False,
    ) -> None:
        from oncolearn.trainer import OncoTrainer

        self._trainer = OncoTrainer(config)
        self._extra_callbacks = extra_callbacks or []
        self._config = config
        self._cv_mode = cv_mode

    def train(self) -> Dict:
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary

        t = self._config.training
        out = self._config.output
        output_dir = Path(out.dir) / out.experiment_name

        if self._cv_mode:
            callbacks: List[pl.Callback] = [
                ModelSummary(max_depth=1),
                *self._extra_callbacks,
            ]
        else:
            callbacks = [
                ModelSummary(max_depth=1),
                ModelCheckpoint(
                    dirpath=str(output_dir),
                    filename="best_model",
                    monitor="val_acc",
                    mode="max",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor="val_acc",
                    patience=t.early_stopping_patience,
                    mode="max",
                ),
                *self._extra_callbacks,
            ]

        gradient_clip_val = t.regularization.gradient_clip_val or None
        pl_trainer = pl.Trainer(
            max_epochs=t.max_epochs,
            accelerator=t.accelerator,
            devices=t.devices,
            default_root_dir=str(output_dir),
            callbacks=callbacks,
            log_every_n_steps=1,
            gradient_clip_val=gradient_clip_val,
            enable_progress_bar=True,
            enable_model_summary=False,
            limit_val_batches=0.0 if self._cv_mode else 1.0,
            num_sanity_val_steps=0 if self._cv_mode else 2,
        )
        pl_trainer.fit(
            self._trainer.model, datamodule=self._trainer.datamodule
        )
        metrics = dict(pl_trainer.callback_metrics)
        if self._cv_mode:
            test_results = pl_trainer.test(
                self._trainer.model, datamodule=self._trainer.datamodule, verbose=False
            )
            if test_results:
                metrics.update(test_results[0])
        return metrics


# ---------------------------------------------------------------------------
# Pruning callback helper
# ---------------------------------------------------------------------------

def _make_pruning_callback(
    trial: optuna.Trial, metric: str
) -> Optional[pl.Callback]:
    try:
        from optuna.integration import PyTorchLightningPruningCallback

        return PyTorchLightningPruningCallback(trial, monitor=metric)
    except ImportError:
        logger.debug(
            "optuna-integration not installed; trial pruning disabled."
        )
        return None
