from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import pandas as pd

class BaseTabularParser(ABC):
    """
    Abstract Base Class for all Tabular Data Parsers.
    Guarantees an extensible protocol for `TabularDataModule`.
    """
    
    @classmethod
    @abstractmethod
    def can_parse(cls, file_path: Path) -> bool:
        """
        Returns True if this parser can process the given file/structure.
        """
        pass
        
    @classmethod
    @abstractmethod
    def parse(cls, file_path: Path) -> pd.DataFrame:
        """
        Loads the tabular data at the specified path and returns a normalized
        pandas DataFrame.
        """
        pass
