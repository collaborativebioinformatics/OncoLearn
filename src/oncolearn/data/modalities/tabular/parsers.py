import pandas as pd
from typing import List

class BaseTabularParser:
    def can_parse(self, file_path) -> bool:
        return False
    def parse(self, file_path) -> pd.DataFrame:
        raise NotImplementedError

class TSVParser(BaseTabularParser):
    def can_parse(self, file_path) -> bool:
        return str(file_path).endswith('.tsv') or str(file_path).endswith('.txt')
        
    def parse(self, file_path) -> pd.DataFrame:
        df = pd.read_csv(file_path, sep='\t')
        
        # If the first column is Unnamed: 0, it's probably the patient ID
        if df.columns[0] == 'Unnamed: 0':
            df.rename(columns={'Unnamed: 0': 'patient_id'}, inplace=True)
            
        # Clean up TCGA IDs up to 12 chars if needed
        if 'patient_id' in df.columns:
            df['patient_id'] = df['patient_id'].apply(lambda x: x[:12] if isinstance(x, str) and x.startswith('TCGA') else x)
            
        # Handle label mapped correctly, BRCA-data-top1000 uses Subtype
        if 'Subtype' in df.columns:
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            # Rename to match standard expectation or just let dataset map
            df['label'] = encoder.fit_transform(df['Subtype'])
            df.drop(columns=['Subtype'], inplace=True)
            
        return df

DEFAULT_PARSERS = [TSVParser()]
