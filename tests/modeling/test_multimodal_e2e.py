import os
import pytest
import torch
import pytorch_lightning as pl
from oncolearn.data.modalities.tabular import TabularDataModule
from oncolearn.data.modalities.image import ImageDataModule
from oncolearn.data.multimodal import MultimodalDataModule
from oncolearn.modeling.fusion import GatedLateFusionClassifier, GatedLateFusionLightning

pytest.importorskip("pydicom")

@pytest.mark.skipif(not os.path.exists("data/xenabrowser/TCGA-BRCA") or not os.path.exists("data/tcia/TCGA-BRCA"), reason="Requires local Multimodal data")
def test_multimodal_e2e():
    # 1. Instantiate Modalities explicitly to avoid registry magic failures in simple tests
    # (Though we could use registry strings here directly if we wanted)
    dm_tabular = TabularDataModule(
        cohort_code="TCGA-BRCA", 
        data_dir="data/xenabrowser",
        batch_size=2,
        num_workers=0,
        label_column=None,
        train_split=1.0,
        features_files=["TCGA-BRCA.clinical.tsv"]
    )
    
    dm_image = ImageDataModule(
        tcia_manifest_url=None,
        tcia_cohort_name="BRCA",
        image_size=(224, 224),
        batch_size=2,
        num_workers=0,
        train_split=1.0,
        data_dir="data/tcia"
    )
    
    # 2. Build the Multimodal pipeline merging on patient_id
    mm_data = MultimodalDataModule(
        modalities=[dm_tabular, dm_image],
        join_on="patient_id",
        strategy="inner"
    )
    
    mm_data.setup()
    
    if len(mm_data.train_dataset) == 0:
        pytest.skip("No intersecting multimodal data loaded. Check if the TCIA and Xenabrowser IDs overlap in your local subset.")
        
    batch_0 = mm_data.train_dataset[0]
    input_dim_tabular = batch_0["tabular"].shape[0]
    
    # 3. Instantiate dummy encoders explicitly
    # In a full run, we would map the Registry strings
    gene_encoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim_tabular, 128),
        torch.nn.ReLU()
    )
    
    # Dummy 3D ViT for image (Pool N frames, then embed)
    class DummyImageEncoder(torch.nn.Module):
        def forward(self, x_seq, ids):
            B, N, C, H, W = x_seq.shape
            # Return static embedding (B, 256)
            return torch.zeros((B, 256), device=x_seq.device)
            
    img_encoder = DummyImageEncoder()
    
    fusion_model = GatedLateFusionClassifier(
        gene_encoder=gene_encoder,
        clinical_encoder=None,
        image_encoder=img_encoder,
        gene_dim=128,
        image_dim=256,
        num_stage_classes=2,
        num_subtype_classes=0
    )
    
    pl_model = GatedLateFusionLightning(model=fusion_model)
    
    trainer = pl.Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False)
    
    trainer.fit(pl_model, datamodule=mm_data)
