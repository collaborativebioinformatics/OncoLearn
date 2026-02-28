import os
import pytest
import torch
from torch import nn
import pytorch_lightning as pl
from oncolearn.data.modalities.tabular import TabularDataModule

class DummyTabularModel(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, 2)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.layer(x)
        
    def training_step(self, batch, batch_idx):
        x = batch["tabular"]
        y = batch.get("label", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        loss = self.loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

@pytest.mark.skipif(not os.path.exists("data/xenabrowser/TCGA-BRCA"), reason="Requires local Tabular data")
def test_tabular_e2e():
    # Explicitly use data/xenabrowser which already has TCGA-BRCA
    dm = TabularDataModule(
        cohort_code="TCGA-BRCA", 
        data_dir="data/xenabrowser",
        batch_size=4,
        num_workers=0,
        label_column=None,
        features_files=["TCGA-BRCA.clinical.tsv"]
    )
    
    # Skip prepare_data to avoid download triggers, we have local data
    dm.setup()
    
    assert len(dm.train_dataset) > 0, "No tabular data loaded."
        
    input_dim = dm.train_dataset[0]["tabular"].shape[0]
    
    model = DummyTabularModel(input_dim=input_dim)
    trainer = pl.Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False)
    
    trainer.fit(model, datamodule=dm)
