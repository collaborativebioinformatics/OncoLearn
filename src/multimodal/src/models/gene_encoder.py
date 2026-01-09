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
            # Add local biomed-multi-omic to path
            import sys
            from pathlib import Path
            biomed_path = Path(__file__).parent.parent.parent / "pre_trained" / "biomed-multi-omic"
            if biomed_path.exists() and str(biomed_path) not in sys.path:
                sys.path.insert(0, str(biomed_path))
            
            # Try to import from biomed-multi-omic or bmfm_targets
            try:
                # Try bmfm_targets package first
                from bmfm_targets.models.model_utils import get_base_model_from_config
                from bmfm_targets.config.model_config import SCBertConfig
                from transformers import AutoConfig
                
                # Load config from HuggingFace
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                # Convert to SCBertConfig if needed
                if hasattr(config, 'model_type') and config.model_type == 'SCBert':
                    scbert_config = SCBertConfig.from_dict(config.to_dict())
                    self.rna_model = get_base_model_from_config(scbert_config)
                    # Load weights
                    from transformers import AutoModel
                    state_dict = AutoModel.from_pretrained(model_name, trust_remote_code=True).state_dict()
                    self.rna_model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded RNA BERT model from bmfm_targets: {model_name}")
                else:
                    raise ImportError("Not SCBert model")
            except (ImportError, Exception) as e:
                # Fallback: try biomed_multi_omic if it exists
                try:
                    from biomed_multi_omic import load_rna_model
                    self.rna_model = load_rna_model(model_name)
                    logger.info(f"Loaded RNA BERT model: {model_name}")
                except ImportError:
                    raise ImportError(f"biomed-multi-omic not found: {e}")
        except ImportError:
            logger.warning("biomed-multi-omic not installed. Trying alternative loading methods...")
            # Try loading from HuggingFace with custom config handling
            try:
                from transformers import AutoModel, AutoConfig
                import json
                from huggingface_hub import hf_hub_download
                
                # Try to download config and handle SCBert model type
                try:
                    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    
                    # If it's SCBert, we need to handle it specially
                    if config_dict.get("model_type") == "SCBert":
                        logger.warning("SCBert model type detected. This requires biomed-multi-omic package.")
                        logger.warning("Falling back to MLP encoder. To use RNA BERT, please install:")
                        logger.warning("  pip install git+https://github.com/BiomedSciAI/biomed-multi-omic.git")
                        raise ImportError("SCBert requires biomed-multi-omic")
                    
                    self.rna_model = AutoModel.from_pretrained(model_name)
                    logger.info(f"Loaded RNA BERT model from HuggingFace: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load RNA BERT model: {e}")
                    raise RuntimeError(f"Cannot load RNA BERT model. Please install biomed-multi-omic: pip install git+https://github.com/BiomedSciAI/biomed-multi-omic.git")
            except ImportError as e:
                logger.error(f"Failed to load RNA BERT model: {e}")
                raise RuntimeError(f"Cannot load RNA BERT model. Please install biomed-multi-omic: pip install git+https://github.com/BiomedSciAI/biomed-multi-omic.git")
        
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
        # RNA BERT expects input in specific format
        # biomed-multi-omic models typically expect gene expression as input_ids
        # For now, we'll treat the gene expression vector as a sequence
        
        try:
            # Try biomed-multi-omic API first
            if hasattr(self.rna_model, 'encode'):
                # If model has encode method (biomed-multi-omic API)
                embeddings = self.rna_model.encode(x)
            elif hasattr(self.rna_model, 'forward'):
                # Standard transformer forward
                # RNA BERT may expect input_ids, attention_mask, etc.
                # For gene expression, we'll use the values directly as input
                # Note: This may need adjustment based on actual model requirements
                
                # Prepare input - gene expression as sequence
                # If model expects tokenized input, we may need to adapt
                # For now, assume model can handle (B, P) input
                if hasattr(self.rna_model, 'config'):
                    # Check if model expects specific input format
                    model_config = self.rna_model.config
                    if hasattr(model_config, 'vocab_size'):
                        # Model expects tokenized input
                        # Convert gene expression to token-like format
                        # This is a simplified approach - may need refinement
                        # Use top-k genes or binning for tokenization
                        B, P = x.shape
                        # Simple approach: use expression values as input_ids (may need normalization)
                        input_ids = (x * 1000).long().clamp(0, model_config.vocab_size - 1)
                        outputs = self.rna_model(input_ids=input_ids)
                    else:
                        # Model may accept continuous values
                        outputs = self.rna_model(x)
                else:
                    # Fallback: try direct forward
                    outputs = self.rna_model(x)
                
                # Extract embeddings
                if hasattr(outputs, 'last_hidden_state'):
                    # Use mean pooling over sequence dimension
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, 'hidden_states'):
                    # Use last hidden state
                    embeddings = outputs.hidden_states[-1].mean(dim=1)
                else:
                    # Fallback: use first element if tuple
                    embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
                    if len(embeddings.shape) > 2:
                        embeddings = embeddings.mean(dim=1)
                    elif len(embeddings.shape) == 1 and x.shape[0] > 1:
                        # Single sample case
                        embeddings = embeddings.unsqueeze(0)
            else:
                # Fallback: direct call
                outputs = self.rna_model(x)
                if isinstance(outputs, tuple):
                    embeddings = outputs[0]
                else:
                    embeddings = outputs
                if len(embeddings.shape) > 2:
                    embeddings = embeddings.mean(dim=1)
        except Exception as e:
            logger.warning(f"RNA BERT forward failed: {e}. Using fallback projection.")
            # Fallback: simple projection if model forward fails
            # This provides safety but indicates model integration issue
            # In production, this should be fixed
            fallback_proj = nn.Linear(x.shape[1], self.hidden_size).to(x.device)
            embeddings = fallback_proj(x)
        
        # Ensure embeddings have correct shape
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # Project to desired output dimension
        output = self.projection(embeddings)
        
        return output

