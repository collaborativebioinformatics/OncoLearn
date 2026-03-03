"""
Gene expression encoders: MLP and RNA BERT.
"""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RNABERTEncoder(nn.Module):
    """
    RNA BERT encoder using IBM Research's biomed.rna.bert.110m.mlm.multitask.v1 model.
    
    Wrapper around biomed-multi-omic RNA BERT model for gene expression encoding.
    """
    
    def __init__(
        self,
        model_name: str = "ibm-research/biomed.rna.bert.110m.mlm.multitask.v1",
        output_dim: int = 128,
        freeze_backbone: bool = True,
        device: str = None
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        
        # Try to load biomed-multi-omic model
        try:
            import json
            from huggingface_hub import hf_hub_download
            from bmfm_targets.config.model_config import SCBertConfig
            from bmfm_targets.models.model_utils import get_base_model_from_config
            from bmfm_targets.models.model_utils import register_configs_and_models
            from bmfm_targets.models.model_utils import download_ckpt_from_huggingface
            from bmfm_targets.training.serialization import prepare_model_dict_from_checkpoint
            
            register_configs_and_models()

            # Prefer local cache to avoid a remote round-trip when already downloaded.
            def _hub_download(repo_id, filename):
                try:
                    return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
                except Exception:
                    return hf_hub_download(repo_id=repo_id, filename=filename)

            config_path = _hub_download(model_name, "config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            if "fields" not in config_dict:
                config_dict["fields"] = []
            if "label_columns" not in config_dict:
                config_dict["label_columns"] = []

            scbert_config = SCBertConfig.from_dict(config_dict)
            self.rna_model = get_base_model_from_config(scbert_config)

            # Use local cache when available to skip the remote "Fetching" round-trip.
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
            self.rna_model.load_state_dict(state_dict, strict=False)
            
            logger.info(f"Loaded RNA BERT model from HuggingFace via bmfm_targets: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load RNA BERT model. {e}")
            raise RuntimeError(f"Cannot load RNA BERT model: {e}. Ensure biomed-multi-omic is correctly installed.")
        
        # Get model output dimension
        try:
            # Try to get hidden size from model config
            if hasattr(self.rna_model, 'config'):
                hidden_size = self.rna_model.config.hidden_size
            elif hasattr(self.rna_model, 'hidden_size'):
                hidden_size = self.rna_model.hidden_size
            else:
                # Default for 110M model
                hidden_size = 768
        except:
            hidden_size = 768
        
        self.hidden_size = hidden_size

        # Detect model type to pick the right forward strategy.
        # SCBertConfig has no vocab_size and uses field-based tokenization which
        # requires a non-empty 'fields' list.  Since we inject fields=[] (the
        # pretrained config carries no field definitions), we must use the
        # inputs_embeds path: project each gene's scalar value → hidden_size and
        # pass as pre-computed embeddings so SCEmbeddingsLayer returns them as-is.
        _config = getattr(self.rna_model, 'config', None)
        _model_type = getattr(_config, 'model_type', '').lower() if _config else ''
        self._use_embeds_projection = (_model_type == 'scbert')
        if self._use_embeds_projection:
            # Projects a scalar expression value per gene → hidden_size vector.
            self._input_proj = nn.Linear(1, self.hidden_size)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.rna_model.parameters():
                param.requires_grad = False
            logger.info("RNA BERT backbone frozen")

        # Projection layer to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P) gene expression values (raw counts or normalized)

        Returns:
            (B, output_dim) gene embedding
        """
        try:
            if hasattr(self.rna_model, 'encode'):
                # biomed-multi-omic API
                embeddings = self.rna_model.encode(x)
            elif hasattr(self.rna_model, 'forward'):
                if self._use_embeds_projection:
                    # SCBert: project each gene's scalar expression value to hidden_size
                    # and pass as inputs_embeds.  SCEmbeddingsLayer returns inputs_embeds
                    # directly when provided, bypassing the field-based tokenization that
                    # requires a non-empty 'fields' list in the config.
                    # x: (B, P) → (B, P, 1) → (B, P, hidden_size)
                    inputs_embeds = self._input_proj(x.unsqueeze(-1))
                    outputs = self.rna_model(inputs_embeds=inputs_embeds)
                else:
                    outputs = self.rna_model(x)

                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output  # (B, hidden_size)
                elif hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state.mean(dim=1)  # (B, hidden_size)
                elif hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, 'hidden_states'):
                    embeddings = outputs.hidden_states[-1].mean(dim=1)
                else:
                    embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
                    if len(embeddings.shape) > 2:
                        embeddings = embeddings.mean(dim=1)
            else:
                outputs = self.rna_model(x)
                embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
                if len(embeddings.shape) > 2:
                    embeddings = embeddings.mean(dim=1)
        except Exception as e:
            import traceback
            logger.warning(f"RNA BERT forward failed: {e}.\n{traceback.format_exc()}\nUsing fallback projection.")
            if not hasattr(self, 'fallback_proj'):
                self.fallback_proj = nn.Linear(x.shape[1], self.hidden_size).to(x.device)
            embeddings = self.fallback_proj(x)

        # Ensure embeddings have correct shape
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)

        # Project to desired output dimension
        output = self.projection(embeddings)

        return output

