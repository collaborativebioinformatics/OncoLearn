"""
Pipeline loader utilities: load_pipeline_file and _make_reader.

These functions have no pytorch_lightning dependency and can be imported in
pure-Python / test environments.
"""
import importlib.util
from pathlib import Path

from .nodes import Dataset, Load, TabularModality, Sequence
from .readers.base import BaseReader


def _make_reader(modality: TabularModality) -> BaseReader:
    """Instantiate the appropriate reader for a Modality's pipeline source."""
    pipeline = modality.pipeline
    # Find the first Load node to extract the DataSource
    if isinstance(pipeline, Load):
        source = pipeline.source
    elif isinstance(pipeline, Sequence):
        load_sources = [step.source for step in pipeline.steps if isinstance(step, Load)]
        if not load_sources:
            raise ValueError(
                f"Modality '{modality.name}': Sequence contains no Load nodes."
            )
        if any(s != load_sources[0] for s in load_sources[1:]):
            raise ValueError(
                f"Modality '{modality.name}': all Load nodes in a Sequence must use "
                f"the same DataSource (mixed sources are not supported)."
            )
        source = load_sources[0]
    else:
        raise ValueError(
            f"Modality '{modality.name}': unknown pipeline type {type(pipeline)}"
        )

    reader_type = source.reader
    if reader_type == "auto":
        reader_type = "cbioportal" if "cbioportal" in source.config.lower() else "xenabrowser"

    if reader_type == "cbioportal":
        from .readers.cbioportal import CbioPortalReader
        return CbioPortalReader(source.config, source.base_dir)
    if reader_type == "xenabrowser":
        from .readers.xenabrowser import XenabrowserReader
        return XenabrowserReader(source.config, source.base_dir)

    raise ValueError(f"Unknown reader type: '{reader_type}'")


def load_pipeline_file(path: str) -> Dataset:
    """Import a pipeline ``.py`` file and return its ``dataset`` attribute.

    The file must define a module-level ``dataset`` variable of type
    :class:`Dataset`.

    Args:
        path: Path to the pipeline Python file.

    Returns:
        The :class:`Dataset` node defined in the file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        AttributeError: If the file does not define a ``dataset`` attribute.
        TypeError: If ``dataset`` is not a :class:`Dataset` instance.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pipeline file not found: {p}")

    spec = importlib.util.spec_from_file_location("_oncolearn_pipeline", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "dataset"):
        raise AttributeError(
            f"Pipeline file '{path}' must define a top-level 'dataset' variable "
            f"of type Dataset."
        )
    dataset = module.dataset
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Pipeline file '{path}': 'dataset' must be a Dataset instance, "
            f"got {type(dataset)}"
        )
    return dataset
