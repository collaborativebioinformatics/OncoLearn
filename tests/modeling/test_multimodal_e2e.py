"""
End-to-end multimodal pipeline smoke test.

Requires real local data (TCIA + Xenabrowser downloads) so is always skipped
in automated test runs.  Run manually when data is available.
"""
import os
import pytest
import pytorch_lightning as pl

from oncolearn.data.modalities.tabular.gene import GeneDataModule
from oncolearn.data.modalities.image.dataset import ImageDataModule
from oncolearn.data.multimodal import MultimodalDataModule
from oncolearn.config import load_config


@pytest.mark.skipif(
    not os.path.exists("data/xenabrowser/TCGA-BRCA")
    or not os.path.exists("data/tcia/TCGA-BRCA")
    # Checkpoint is mounted at /workspace/models inside Docker; skip on host.
    or not os.path.exists("/workspace/models/breast_MR_checkpoint.pth.tar"),
    reason="Requires local multimodal data (TCIA + Xenabrowser) and FM-BCMRI checkpoint at /workspace/models/",
)
def test_multimodal_e2e():
    cfg = load_config("data/configs/tcga_brca_multimodal.yaml")

    dm_gene = GeneDataModule(
        cohort_code="TCGA-BRCA",
        base_directory="data/xenabrowser",
        files=["TCGA-BRCA.mirna.tsv", "pam50.tsv"],
        batch_size=2,
        num_workers=0,
        batch_key="oncolearn.modality.gene",
    )
    dm_gene.name = "oncolearn.modality.gene"

    dm_image = ImageDataModule(
        tcia_cohort_name="BRCA",
        base_directory="data/tcia",
        batch_size=2,
        num_workers=0,
        batch_key="oncolearn.modality.image",
    )
    dm_image.name = "oncolearn.modality.image"

    mm_data = MultimodalDataModule(
        modalities=[dm_gene, dm_image],
        join_on="patient_id",
        strategy="inner",
        batch_size=2,
        num_workers=0,
    )
    mm_data.setup()

    if len(mm_data.train_dataset) == 0:
        pytest.skip(
            "No intersecting multimodal patients found. "
            "Check that TCIA and Xenabrowser IDs overlap in your local data."
        )

    from oncolearn.registry import get_model
    import oncolearn.modeling  # noqa: F401
    model_cls = get_model(cfg.model.name)
    model = model_cls(cfg)

    trainer = pl.Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False)
    trainer.fit(model, datamodule=mm_data)
