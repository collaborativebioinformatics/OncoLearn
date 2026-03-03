"""
Base encoder class for OncoLearn encoders.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig

logger = logging.getLogger(__name__)


class BaseEncoder(nn.Module):
    """
    Base encoder with common HuggingFace model loading and utility methods.

    When ``huggingface_models`` is non-empty each model name is loaded via
    ``_load_huggingface_model`` (defaults to ``AutoModel.from_pretrained``) and
    registered as ``self.hf_enc1``, ``self.hf_enc2``, etc.  Subclasses may
    override ``_load_huggingface_model`` for specialised loading (e.g. bmfm_targets).

    Args:
        config: Full experiment config.
        output_dim: Output embedding dimension for this encoder.
        huggingface_models: Optional list of HuggingFace model IDs to load.
    """

    def __init__(
        self,
        config: "OncoLearnConfig",
        output_dim: int = 128,
        huggingface_models: list[str] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.freeze_encoders: bool = config.model.freeze_encoders

        if huggingface_models:
            for i, model_name in enumerate(huggingface_models, start=1):
                enc = self._load_huggingface_model(model_name)
                if self.freeze_encoders:
                    self._freeze(enc)
                    logger.info(f"Frozen HuggingFace encoder: {model_name}")
                setattr(self, f"hf_enc{i}", enc)

    def _load_huggingface_model(self, model_name: str) -> nn.Module:
        """Load a HuggingFace model via ``AutoModel.from_pretrained``.

        Override in subclasses for custom loading behaviour.
        """
        from transformers import AutoModel

        logger.info(f"Loading HuggingFace model: {model_name}")
        return AutoModel.from_pretrained(model_name)

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        """Freeze all parameters of *module* in-place."""
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def _pool_hf_outputs(outputs, strategy: str = "mean") -> torch.Tensor:
        """Extract a fixed-size embedding from HuggingFace model outputs.

        Args:
            outputs: HuggingFace ``ModelOutput``, tuple, or raw tensor.
            strategy: ``"cls"`` returns the [CLS] token (index 0);
                      ``"mean"`` mean-pools over the sequence dimension.

        Returns:
            Tensor of shape ``(B, hidden_size)``.
        """
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        elif isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        if hidden.dim() == 2:
            return hidden
        return hidden[:, 0] if strategy == "cls" else hidden.mean(dim=1)
