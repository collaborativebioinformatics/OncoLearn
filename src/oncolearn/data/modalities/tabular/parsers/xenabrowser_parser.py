import pandas as pd
from pathlib import Path
from .base import BaseTabularParser


class XenabrowserParser(BaseTabularParser):
    """
    Tabular Parser specialized in loading xenabrowser TSVs with gene expressions.
    """
    
    @classmethod
    def can_parse(cls, file_path: Path) -> bool:
        # Check standard extension and naming convention or custom flag
        # We assume for this implementation that if it's a TSV in the xenabrowser output dir, we can parse it.
        return file_path.suffix.lower() == '.tsv'
        
    @classmethod
    def parse(cls, file_path: Path) -> pd.DataFrame:
        try:
            # Low memory off due to large column counts in xenabrowser
            df = pd.read_csv(str(file_path), sep='\t', low_memory=False)
            
            # Common Xenabrowser tables have 'sample' as the patient ID column.
            # We rename it to the standard internal key "patient_id" if it exists.
            if 'sample' in df.columns:
                df = df.rename(columns={'sample': 'patient_id'})
                
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to parse Xenabrowser TSV at {file_path}: {e}")
