"""
End-to-end multimodal pipeline smoke test using PipelineDataModule + OncoTrainer.

Requires real local data (TCIA + Xenabrowser downloads) so is always skipped
in automated test runs.  Run manually when data is available.
"""
import os
import pytest
import pytorch_lightning as pl

from oncolearn.data.modules.multimodal import MultimodalDataModule
from oncolearn.config import load_config


@pytest.mark.skipif(
    not os.path.exists("data/sources/xenabrowser/TCGA-BRCA"),
    reason="Requires local Xenabrowser data",
)
def test_multimodal_e2e():
    from oncolearn.data.pipeline import DataSource, Load, Modality
    from oncolearn.data.modules.base import PipelineDataModule
    from oncolearn.data.pipeline.transforms import map_ajcc_stage

    ds = DataSource(config="xenabrowser", base_dir="data/sources/xenabrowser/TCGA-BRCA")

    dm_gene = PipelineDataModule.from_modality(
        Modality(name="oncolearn.modality.gene", pipeline=Load("TCGA-BRCA.mirna.tsv", source=ds)),
        batch_size=2,
        num_workers=0,
    )
    dm_clinical = PipelineDataModule.from_modality(
        Modality(
            name="oncolearn.modality.clinical",
            pipeline=Load("TCGA-BRCA.clinical.tsv", source=ds),
            label_col="ajcc_pathologic_stage.diagnoses",
            label_transform=map_ajcc_stage,
        ),
        batch_size=2,
        num_workers=0,
    )

    mm_data = MultimodalDataModule(
        modalities=[dm_gene, dm_clinical],
        join_on="patient_id",
        strategy="inner",
        batch_size=2,
        num_workers=0,
    )
    mm_data.setup()

    if len(mm_data.train_dataset) == 0:
        pytest.skip("No intersecting multimodal patients found.")

    cfg = load_config("data/configs/modeling/multimodal/tcga_brca_multimodal.yaml")

    from oncolearn.registry import get_model
    import oncolearn.modeling  # noqa: F401
    model_cls = get_model(cfg.model.name)
    model = model_cls(cfg)

    trainer = pl.Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False)
    trainer.fit(model, datamodule=mm_data)
