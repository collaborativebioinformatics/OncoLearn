from typing import List, Optional
import pytorch_lightning as pl

from oncolearn.registry import get_model
from oncolearn.data.multimodal import MultimodalDataModule


class OncoTrainer:
    """
    A builder pipeline that instantiates and executes a PyTorch Lightning Training loop
    given user-requested modalities and registered model names.
    """
    
    def __init__(
        self,
        modalities: List[str],
        model_name: str,
        max_epochs: int = 10,
        accelerator: str = "auto",
        devices: int = 1,
        join_on: str = "patient_id",
        strategy: str = "inner",
        model_kwargs: dict = None,
        data_kwargs: dict = None
    ):
        """
        Args:
            modalities: List of registered modality names (e.g., ["image", "tabular"])
            model_name: Name of the registered LightningModule
            max_epochs: Epoch limit for the PyTorch Lightning trainer
            accelerator: Hardware accelerator ('cpu', 'gpu', 'mps', 'auto')
            devices: Number of devices
            join_on: Key used by MultimodalDataModule to align patients
            strategy: Join strategy (e.g. 'inner')
            model_kwargs: Dictionary of arguments passed directly to the model constructor
            data_kwargs: Dictionary of arguments configuring the MultimodalDataModule batches
        """
        self.modalities_requested = modalities
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        
        # 1. Initialize Multimodal DataModule Builder (fetches from registry implicitly)
        self.datamodule = MultimodalDataModule(
            modalities=self.modalities_requested,
            join_on=join_on,
            strategy=strategy,
            **(data_kwargs or {})
        )
        
        # 2. Fetch and initialize Lightning Module
        model_cls = get_model(self.model_name)
        
        # Validate that the requested modalities cover the model's required endpoints
        if hasattr(model_cls, "expected_modalities") and model_cls.expected_modalities:
            missing = set(model_cls.expected_modalities) - set(self.modalities_requested)
            if missing:
                print(f"Warning: Model {self.model_name} expects modalities {model_cls.expected_modalities} but {missing} is missing.")
                
        self.model = model_cls(**(model_kwargs or {}))
        
        # 3. Setup core PyTorch Lightning Trainer
        self.pl_trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            enable_progress_bar=True,
            logger=True,
            log_every_n_steps=10
        )
        
    def train(self):
        """
        Orchestrate the fitting process using PyTorch Lightning.
        """
        print(f"Starting OncoTrainer execution for Model: {self.model_name} | Modalities: {self.modalities_requested}")
        self.pl_trainer.fit(model=self.model, datamodule=self.datamodule)
        
    def test(self):
        """
        Evaluate the pipeline using PyTorch Lightning test loop.
        """
        self.pl_trainer.test(model=self.model, datamodule=self.datamodule)
