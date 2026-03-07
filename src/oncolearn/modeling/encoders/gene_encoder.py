"""
Gene expression encoder: RNA BERT (IBM biomed-multi-omic).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

from oncolearn.registry import register_config, register_encoder
from .base import BaseEncoder

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig

logger = logging.getLogger(__name__)


@register_config("gene")
@dataclass
class RNABERTEncoderConfig:
    """Configuration for the RNA BERT gene expression encoder."""

    output_dim: int = 128
    huggingface_models: Optional[List[str]] = None
    device: Optional[str] = None
    max_seq_len: int = 512
    projection_dropout: float = 0.1


@register_encoder("gene")
class RNABERTEncoder(BaseEncoder):
    """
    RNA BERT encoder using IBM Research's biomed.rna.bert.110m.mlm.multitask.v1 model.

    Wrapper around biomed-multi-omic RNA BERT model for gene expression encoding.
    Overrides ``_load_huggingface_model`` to use bmfm_targets instead of AutoModel.

    Args:
        config: Full experiment config.  Encoder-specific parameters are read via
                :func:`~oncolearn.registry.resolve_encoder_config` which merges
                :class:`RNABERTEncoderConfig` defaults with any YAML overrides.
    """

    def __init__(self, config: "OncoLearnConfig") -> None:
        from oncolearn.registry import resolve_encoder_config

        enc_cfg: RNABERTEncoderConfig = resolve_encoder_config(
            type(self), "gene", config
        )

        if enc_cfg.huggingface_models is None:
            enc_cfg.huggingface_models = [
                "ibm-research/biomed.rna.bert.110m.mlm.multitask.v1"
            ]

        # _load_huggingface_model override is resolved before super().__init__ runs,
        # so hf_enc1 will be loaded via bmfm_targets.
        super().__init__(
            config,
            output_dim=enc_cfg.output_dim,
            huggingface_models=enc_cfg.huggingface_models,
        )

        # If loading failed, hf_enc1 is nn.Identity — mark for pure-linear fallback.
        self._rna_bert_unavailable = isinstance(self.hf_enc1, nn.Identity)

        rna_model = self.hf_enc1

        # Get model hidden size
        try:
            if hasattr(rna_model, "config"):
                hidden_size = rna_model.config.hidden_size
            elif hasattr(rna_model, "hidden_size"):
                hidden_size = rna_model.hidden_size
            else:
                hidden_size = 768
        except Exception:
            hidden_size = 768

        self.hidden_size = hidden_size

        # Detect SCBert model type to pick the right forward strategy
        _cfg = getattr(rna_model, "config", None)
        _model_type = getattr(_cfg, "model_type", "").lower() if _cfg else ""
        self._use_embeds_projection = _model_type == "scbert"
        if self._use_embeds_projection:
            self._input_proj = nn.Linear(1, self.hidden_size)

        # Truncate input to top-N features before passing to SCBert.
        # SCBert attention is O(n²) — 1881 tokens × 12 layers × 12 heads exceeds RAM.
        # We keep the top max_seq_len features by absolute magnitude (highest-expressed
        # miRNAs carry the most signal).
        self.max_seq_len = enc_cfg.max_seq_len

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, enc_cfg.output_dim),
            nn.LayerNorm(enc_cfg.output_dim),
            nn.GELU(),
            nn.Dropout(enc_cfg.projection_dropout),
        )

    def _load_huggingface_model(self, model_name: str) -> nn.Module:
        """Load RNA BERT via bmfm_targets instead of AutoModel."""
        try:
            import json
            from huggingface_hub import hf_hub_download
            from bmfm_targets.config.model_config import SCBertConfig
            from bmfm_targets.models.model_utils import get_base_model_from_config
            from bmfm_targets.models.model_utils import register_configs_and_models
            from bmfm_targets.models.model_utils import download_ckpt_from_huggingface
            from bmfm_targets.training.serialization import prepare_model_dict_from_checkpoint

            register_configs_and_models()

            def _hub_download(repo_id, filename):
                try:
                    return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
                except Exception:
                    return hf_hub_download(repo_id=repo_id, filename=filename)

            config_path = _hub_download(model_name, "config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            if "fields" not in config_dict:
                config_dict["fields"] = []
            if "label_columns" not in config_dict:
                config_dict["label_columns"] = []

            scbert_config = SCBertConfig.from_dict(config_dict)
            model = get_base_model_from_config(scbert_config)

            try:
                from huggingface_hub import snapshot_download
                from pathlib import Path as _Path
                _local = snapshot_download(
                    model_name, ignore_patterns=["*.git*", "*.md*"], local_files_only=True
                )
                _ckpts = list(_Path(_local).glob("*.ckpt"))
                weights_path = str(_ckpts[0]) if _ckpts else download_ckpt_from_huggingface(model_name)
            except Exception:
                weights_path = download_ckpt_from_huggingface(model_name)

            state_dict = prepare_model_dict_from_checkpoint(weights_path)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded RNA BERT model from HuggingFace via bmfm_targets: {model_name}")
            return model
        except Exception as e:
            logger.warning(
                f"Failed to load RNA BERT model ({e}). "
                "Falling back to linear projection — training will continue without pre-trained weights."
            )
            return nn.Identity()  # forward() fallback_proj handles the actual projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P) gene expression values

        Returns:
            (B, output_dim) gene embedding
        """
        # Fast path: RNA BERT unavailable at load time — use linear projection directly.
        if self._rna_bert_unavailable:
            if not hasattr(self, "fallback_proj"):
                self.fallback_proj = nn.Linear(x.shape[1], self.hidden_size).to(x.device)
            embeddings = self.fallback_proj(x)
            return self.projection(embeddings)

        rna_model = self.hf_enc1
        try:
            if hasattr(rna_model, "encode"):
                embeddings = rna_model.encode(x)
            else:
                if self._use_embeds_projection:
                    # Truncate to top-max_seq_len features by magnitude to keep
                    # attention memory O(max_seq_len²) instead of O(n_features²).
                    if x.shape[1] > self.max_seq_len:
                        topk_idx = x.abs().topk(self.max_seq_len, dim=1).indices
                        x_trunc = x.gather(1, topk_idx)
                    else:
                        x_trunc = x
                    inputs_embeds = self._input_proj(x_trunc.unsqueeze(-1))
                    outputs = rna_model(inputs_embeds=inputs_embeds)
                else:
                    outputs = rna_model(x)
                embeddings = self._pool_hf_outputs(outputs, strategy="mean")
        except Exception as e:
            import traceback
            logger.warning(
                f"RNA BERT forward failed: {e}.\n{traceback.format_exc()}\nUsing fallback projection."
            )
            if not hasattr(self, "fallback_proj"):
                self.fallback_proj = nn.Linear(x.shape[1], self.hidden_size).to(x.device)
            embeddings = self.fallback_proj(x)

        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        return self.projection(embeddings)
