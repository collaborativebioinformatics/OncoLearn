import pytest
import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset
from oncolearn.data.multimodal import MultimodalDataset


class MockModalityDataset(Dataset):
    """A dummy PyTorch Dataset implementing get_keys for the builder pattern."""
    def __init__(self, prefix: str, patient_ids: List[str]):
        self.prefix = prefix
        self.patient_ids = patient_ids
        
    def get_keys(self) -> List[str]:
        return self.patient_ids
        
    def __len__(self) -> int:
        return len(self.patient_ids)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            self.prefix: torch.tensor([idx], dtype=torch.float32),
            "patient_id": self.patient_ids[idx],
            "label": 1 if idx % 2 == 0 else 0
        }


def test_multimodal_dataset_inner_join():
    """Test that MultimodalDataset accurately joins patient intersection."""
    # Patient 2 is missing from clinical, Patient 4 is missing from image
    ds_image = MockModalityDataset("image", ["P1", "P2", "P3"])
    ds_clinical = MockModalityDataset("clinical", ["P1", "P3", "P4"])
    
    mm_dataset = MultimodalDataset(
        datasets={"image": ds_image, "clinical": ds_clinical},
        join_on="patient_id",
        strategy="inner"
    )
    
    # Should only contain intersecting patients (P1, P3)
    assert len(mm_dataset) == 2
    
    # Test record contents
    record_0 = mm_dataset[0]
    assert record_0["patient_id"] == "P1"
    assert "image" in record_0
    assert "clinical" in record_0
    assert "label" in record_0
    
    record_1 = mm_dataset[1]
    assert record_1["patient_id"] == "P3"
    assert "label" in record_1


def test_multimodal_dataset_missing_get_keys():
    """Test that a lacking `get_keys` protocol raises an AttributeError."""
    class BadDataset(Dataset):
        def __len__(self): return 1
        def __getitem__(self, idx): return {}
        
    ds_good = MockModalityDataset("image", ["P1"])
    ds_bad = BadDataset()
    
    with pytest.raises(AttributeError, match="must implement"):
        MultimodalDataset(
            datasets={"image": ds_good, "clinical": ds_bad},
            join_on="patient_id",
            strategy="inner"
        )
