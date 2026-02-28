import os
import pytest
import torch
from torch import nn
import pytorch_lightning as pl
from oncolearn.data.modalities.image import ImageDataModule

pytest.importorskip("pydicom")

class DummyImageModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Emulating the MRMGHierarchicalImageEncoder signature but simple
        self.conv = nn.Conv2d(3, 16, 3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x_seq):
        # x_seq: (B, N, C, H, W)
        B, N, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * N, C, H, W)
        feats = self.pool(torch.relu(self.conv(x_flat))) # (B*N, 16, 1, 1)
        feats = feats.view(B, N, -1) # (B, N, 16)
        # Sequence average pool across the N frames
        feats_pooled = feats.mean(dim=1) # (B, 16)
        
        return self.fc(feats_pooled)
        
    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch.get("label", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        loss = self.loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

@pytest.mark.skipif(not os.path.exists("data/tcia/TCGA-BRCA"), reason="Requires local Image data")
def test_image_e2e():
    # Requires data/tcia to exist or will just finish empty gracefully.
    dm = ImageDataModule(
        tcia_manifest_url=None, # Avoid downloads
        tcia_cohort_name="BRCA",
        image_size=(224, 224),
        batch_size=2,
        num_workers=0,
        data_dir="data/tcia"
    )
    
    dm.setup()
    
    assert len(dm.train_dataset) > 0, "No image data loaded."
        
    model = DummyImageModel()
    trainer = pl.Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False)
    
    trainer.fit(model, datamodule=dm)
