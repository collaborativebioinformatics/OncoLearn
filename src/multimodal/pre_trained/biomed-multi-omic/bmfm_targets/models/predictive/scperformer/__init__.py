"""
Performer model inspired by https://github.com/lucidrains/performer-pytorch
and https://github.com/TencentAILabHealthcare/scBERT.
"""
from .modeling_scperformer import (
    SCPerformerForMaskedLM,
    SCPerformerForSequenceClassification,
    SCPerformerForSequenceLabeling,
    SCPerformerForMultiTaskModeling,
)

__all__ = [
    "SCPerformerForMaskedLM",
    "SCPerformerForSequenceClassification",
    "SCPerformerForSequenceLabeling",
    "SCPerformerForMultiTaskModeling",
]
