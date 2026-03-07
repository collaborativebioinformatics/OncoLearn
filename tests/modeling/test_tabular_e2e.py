"""
End-to-end tabular (gene) pipeline smoke test.

Requires real local data so is always skipped in automated test runs.
"""
import os
import pytest
import torch
from torch import nn
import pytorch_lightning as pl

from oncolearn.data.modalities.tabular.gene import GeneDataModule


class DummyTabularModel(pl.LightningModule):
    def __init__(self, input_dim, num_classes=5):
        super().__init__()
        self.layer = nn.Linear(input_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x = batch["gene"]
        y = batch.get("label", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        loss = self.loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


@pytest.mark.skipif(
    not os.path.exists("data/xenabrowser/TCGA-BRCA"),
    reason="Requires local tabular data",
)
def test_tabular_e2e():
    dm = GeneDataModule(
        cohort_code="TCGA-BRCA",
        base_directory="data/xenabrowser",
        files=["TCGA-BRCA.mirna.tsv", "pam50.tsv"],
        batch_size=4,
        num_workers=0,
        batch_key="gene",
    )
    dm.setup()

    assert len(dm.train_dataset) > 0, "No tabular data loaded."

    input_dim = dm.train_dataset[0]["gene"].shape[0]
    # Detect number of classes from labels so CrossEntropyLoss doesn't fail.
    labels = [int(dm.train_dataset[i].get("label", 0)) for i in range(len(dm.train_dataset))]
    num_classes = max(labels) + 1 if labels else 5

    model = DummyTabularModel(input_dim=input_dim, num_classes=num_classes)
    trainer = pl.Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False)
    trainer.fit(model, datamodule=dm)
