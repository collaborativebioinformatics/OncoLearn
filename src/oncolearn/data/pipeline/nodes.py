"""
DSL node dataclasses for the OncoLearn data loading pipeline.
"""
from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union


@dataclass
class BaseModality(ABC):
    """Abstract base for all modality descriptors.

    Attributes:
        name: Dotted modality name used as the batch-routing key
              (e.g. ``"oncolearn.modality.gene"``).  Must match the
              ``modality`` field of the corresponding encoder config.
    """

    name: str


@dataclass
class DataSource:
    """Reference to a data source configuration.

    Attributes:
        config: Path to the source config file, or a label string (e.g.
                ``"data/configs/sources/cbioportal/brca_tcga.yaml"`` or ``"xenabrowser"``).
                Used by the ``"auto"`` reader to select the correct reader type.
        base_dir: Root directory where data files are stored.
        reader: Reader to use.  ``"auto"`` detects from *config*:
                if ``"cbioportal"`` appears in the string → :class:`CbioPortalReader`,
                otherwise → :class:`XenabrowserReader`.
    """

    config: str
    base_dir: str
    reader: str = "auto"


@dataclass
class Load:
    """Pipeline leaf node: load one dataset by name.

    Pushes a single :class:`pandas.DataFrame` onto the execution stack.

    Attributes:
        name: Dataset name looked up in the reader (e.g. ``"clinical_patient"``
              for cBioPortal, or a filename like ``"TCGA-BRCA.mirna.tsv"`` for
              XenaBrowser).
        source: The :class:`DataSource` that provides the reader.
    """

    name: str
    source: DataSource


@dataclass
class Join:
    """Pipeline node: pop two DataFrames, merge, push result.

    Pops the top two DataFrames from the execution stack (right first, then
    left), merges them, and pushes the result.

    Attributes:
        on: Column name to join on.  Defaults to ``"patient_id"``.
        how: pandas merge strategy (``"inner"``, ``"left"``, ``"outer"``).
    """

    on: str = "patient_id"
    how: str = "inner"


@dataclass
class Sequence:
    """Pipeline composite node: execute a list of steps in order.

    Flattened by the executor so nested :class:`Sequence` nodes are also
    supported.

    Attributes:
        steps: Ordered list of :class:`Load`, :class:`Join`, or nested
               :class:`Sequence` nodes.
    """

    steps: List[Union[Load, Join, "Sequence"]] = field(default_factory=list)


@dataclass
class TabularModality(BaseModality):
    """Descriptor for a tabular data modality loaded via the pipeline executor.

    Attributes:
        name: Dotted modality name used as the batch-routing key
              (e.g. ``"oncolearn.modality.clinical"``).  Must match the
              ``modality`` field of the corresponding encoder config.
        pipeline: A :class:`Load` or :class:`Sequence` that produces the
                  modality's :class:`pandas.DataFrame`.
        label_col: Column in the produced DataFrame that contains raw label
                   values.  After ``label_transform`` is applied the column is
                   renamed to ``"label"`` and the original column is dropped.
                   ``None`` means no labels for this modality.
        label_transform: Callable applied element-wise to ``label_col``.
                         Should return an ``int`` or ``None`` (rows with
                         ``None`` are dropped).  Defaults to identity.
        patient_id_col: Column name for patient identifiers.  Defaults to
                        ``"patient_id"``.
    """

    name: str = ""
    pipeline: Optional[Union[Load, Sequence]] = None
    label_col: Optional[str] = None
    label_transform: Optional[Callable] = None
    patient_id_col: str = "patient_id"


@dataclass
class ImageModality(BaseModality):
    """Descriptor for the DICOM imaging modality.

    Unlike :class:`TabularModality`, this node does not use the pipeline
    executor — it is backed directly by
    :class:`~oncolearn.data.modalities.image.ImageDataModule` which scans
    DICOM files from disk.

    Attributes:
        name: Dotted modality name used as the batch-routing key.
              Must match the ``modality`` field of the image encoder config.
        base_dir: Root directory containing TCIA cohort subdirectories.
        cohort_code: TCIA cohort code (e.g. ``"BRCA"``).
        n_slices: Number of slices to uniformly sample per series.
        prefer_mr: When True, MR series are preferred over MG for each patient.
    """

    name: str = "oncolearn.modality.image"
    base_dir: str = "data/tcia"
    cohort_code: str = "BRCA"
    n_slices: int = 5
    prefer_mr: bool = True


@dataclass
class Dataset:
    """Top-level pipeline descriptor for a complete dataset.

    Attributes:
        modalities: Ordered list of modality nodes.
        name: Optional registry name.  When non-empty,
              ``trainer._build_datamodule`` looks up the dataset class via
              :func:`~oncolearn.registry.get_dataset` instead of using the
              default :class:`~oncolearn.data.datasets.multimodal.MultimodalDataModule`.
        join_on: Patient-ID column used by :class:`MultimodalDataModule` to
                 align samples across modalities.
        join_strategy: Cross-modality join strategy passed to
                       :class:`MultimodalDataModule`.
    """

    modalities: List[BaseModality]
    name: str = ""
    join_on: str = "patient_id"
    join_strategy: str = "inner"
