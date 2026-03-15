"""
End-to-end tabular (gene) pipeline smoke test using PipelineDataModule.

Requires real local data so is always skipped in automated test runs.
"""
import os
import pytest
import torch
from torch import nn
import pytorch_lightning as pl


class DummyTabularModel(pl.LightningModule):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.layer = nn.Linear(input_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x = batch["oncolearn.modality.gene"]
        y = batch.get("label", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        loss = self.loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


@pytest.mark.skipif(
    not os.path.exists("data/sources/xenabrowser/TCGA-BRCA"),
    reason="Requires local tabular data",
)
def test_tabular_e2e():
    from oncolearn.data.pipeline import DataSource, Load, Modality
    from oncolearn.data.modules.base import PipelineDataModule

    ds = DataSource(config="xenabrowser", base_dir="data/sources/xenabrowser/TCGA-BRCA")
    modality = Modality(
        name="oncolearn.modality.gene",
        pipeline=Load("TCGA-BRCA.mirna.tsv", source=ds),
    )
    dm = PipelineDataModule.from_modality(modality, batch_size=4, num_workers=0)
    dm.setup()

    assert len(dm.train_dataset) > 0, "No tabular data loaded."

    input_dim = dm.train_dataset[0]["oncolearn.modality.gene"].shape[0]
    model = DummyTabularModel(input_dim=input_dim, num_classes=4)
    trainer = pl.Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False)
    trainer.fit(model, datamodule=dm)
