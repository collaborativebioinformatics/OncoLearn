"""Abstract base class for pipeline data readers."""
from abc import ABC, abstractmethod

import pandas as pd


class BaseReader(ABC):
    """Protocol for all pipeline data readers.

    A reader knows how to load a named dataset and return a normalized
    :class:`pandas.DataFrame` with a ``patient_id`` column.
    """

    @abstractmethod
    def read(self, name: str) -> pd.DataFrame:
        """Load dataset *name* and return a normalized DataFrame.

        Args:
            name: Dataset identifier (e.g. ``"clinical_patient"`` for
                  cBioPortal, or a filename for XenaBrowser).

        Returns:
            DataFrame with at minimum a ``patient_id`` column.
        """
        ...
