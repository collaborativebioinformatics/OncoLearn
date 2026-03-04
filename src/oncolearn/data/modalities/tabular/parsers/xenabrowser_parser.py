import pandas as pd
from pathlib import Path
from .base import BaseTabularParser


class XenabrowserParser(BaseTabularParser):
    """
    Tabular Parser specialized in loading xenabrowser TSVs with gene expressions.
    Handles both sample-level format (rows=patients) and genomic matrix format
    (rows=genes, cols=patients), transposing the latter automatically.
    """

    @classmethod
    def can_parse(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.tsv'

    @classmethod
    def _is_genomic_matrix(cls, df: pd.DataFrame) -> bool:
        """Return True if columns (after the first) look like TCGA sample IDs."""
        sample_cols = [c for c in df.columns[1:6] if isinstance(c, str) and c.startswith('TCGA-')]
        return len(sample_cols) >= 3

    @classmethod
    def parse(cls, file_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(str(file_path), sep='\t', low_memory=False)

            # Genomic matrix format: rows=features, cols=patients — transpose to rows=patients.
            if cls._is_genomic_matrix(df):
                id_col = df.columns[0]
                df = df.set_index(id_col).T.reset_index()
                df = df.rename(columns={'index': 'patient_id'})
            elif 'sample' in df.columns:
                df = df.rename(columns={'sample': 'patient_id'})
            elif df.columns[0] == 'Unnamed: 0':
                df = df.rename(columns={'Unnamed: 0': 'patient_id'})

            if 'patient_id' in df.columns:
                df['patient_id'] = df['patient_id'].apply(
                    lambda x: x[:12] if isinstance(x, str) and x.startswith('TCGA') else x
                )

            # Encode subtype / PAM50 column as integer label
            label_src = next(
                (c for c in ('Subtype', 'PAM50', 'pam50') if c in df.columns), None
            )
            if label_src:
                from sklearn.preprocessing import LabelEncoder
                # Drop rows with missing or unknown labels — they have no valid supervision signal.
                df = df[df[label_src].notna() & (df[label_src] != 'Unknown')].copy()
                encoder = LabelEncoder()
                df['label'] = encoder.fit_transform(df[label_src])
                df = df.drop(columns=[label_src])

            return df
        except Exception as e:
            raise RuntimeError(f"Failed to parse Xenabrowser TSV at {file_path}: {e}")
