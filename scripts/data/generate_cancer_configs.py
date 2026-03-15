#!/usr/bin/env python3
"""
Script to generate YAML configuration files for TCGA cancer types from Xena Browser.
"""

import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

# Cancer types to process
CANCER_TYPES = [
    ("LAML", "Acute Myeloid Leukemia"),
    ("ACC", "Adrenocortical Cancer"),
    ("CHOL", "Bile Duct Cancer"),
    ("BLCA", "Bladder Cancer"),
    ("BRCA", "Breast Cancer"),
    ("CESC", "Cervical Cancer"),
    ("COAD", "Colon Cancer"),
    ("UCEC", "Endometrioid Cancer"),
    ("ESCA", "Esophageal Cancer"),
    ("GBM", "Glioblastoma"),
    ("HNSC", "Head and Neck Cancer"),
    ("KICH", "Kidney Chromophobe"),
    ("KIRC", "Kidney Clear Cell Carcinoma"),
    ("KIRP", "Kidney Papillary Cell Carcinoma"),
    ("DLBC", "Large B-cell Lymphoma"),
    ("LIHC", "Liver Cancer"),
    ("LGG", "Lower Grade Glioma"),
    ("LUAD", "Lung Adenocarcinoma"),
    ("LUSC", "Lung Squamous Cell Carcinoma"),
    ("SKCM", "Melanoma"),
    ("MESO", "Mesothelioma"),
    ("UVM", "Ocular melanomas"),
    ("OV", "Ovarian Cancer"),
    ("PAAD", "Pancreatic Cancer"),
    ("PCPG", "Pheochromocytoma & Paraganglioma"),
    ("PRAD", "Prostate Cancer"),
    ("READ", "Rectal Cancer"),
    ("SARC", "Sarcoma"),
    ("STAD", "Stomach Cancer"),
    ("TGCT", "Testicular Cancer"),
    ("THYM", "Thymoma"),
    ("THCA", "Thyroid Cancer"),
    ("UCS", "Uterine Carcinosarcoma"),
]


def fetch_dataset_metadata(dataset_id, host="https://gdc.xenahubs.net"):
    """Fetch detailed metadata for a specific dataset."""
    url = f"https://xenabrowser.net/datapages/?dataset={quote(dataset_id)}&host={quote(host)}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        metadata = {}
        
        # Extract metadata using regex patterns
        patterns = {
            'dataset_id': r'dataset ID\s*([^\n]+)',
            'download': r'download\s*([https://][^\s;]+)',
            'samples': r'samples\s*(\d+)',
            'version': r'version\s*([^\n]+)',
            'data_type': r'type of data\s*([^\n]+)',
            'assembly': r'assembly\s*([^\n]+)',
            'unit': r'unit\s*([^\n]+)',
            'platform': r'platform\s*([^\n]+)',
            'gene_mapping': r'gene mapping\s*([https://][^\s]+)',
            'author': r'author\s*([^\n]+)',
            'raw_data': r'raw data\s*([https://][^\s]+)',
            'wrangling': r'wrangling\s*([^\n]+(?:\n(?!(?:input data format|cohort|dataset))[^\n]+)*)',
            'input_data_format': r'input data format\s*([^\n]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Clean up the value
                value = re.sub(r'\s+', ' ', value)
                metadata[key] = value
        
        return metadata
    
    except Exception as e:
        print(f"Error fetching {dataset_id}: {e}")
        return None


def get_cohort_datasets(code, name):
    """Get list of datasets for a cohort from Xena Browser."""
    cohort_name = f"GDC TCGA {name} ({code})"
    url = f"https://xenabrowser.net/datapages/?cohort={quote(cohort_name)}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        # Extract dataset IDs (look for TCGA-{code}.*.tsv patterns)
        dataset_pattern = fr'TCGA-{code}\.[a-z0-9_\-]+\.tsv'
        datasets = list(set(re.findall(dataset_pattern, content)))
        
        print(f"Found {len(datasets)} datasets for {code}")
        return datasets
    
    except Exception as e:
        print(f"Error fetching cohort {code}: {e}")
        return []


def generate_config(code, name):
    """Generate YAML configuration for a cancer type."""
    print(f"\nProcessing {code} - {name}...")
    
    config = {
        'cohort': {
            'code': code,
            'name': f'TCGA-{code}',
            'description': f'TCGA {name} cohort with multi-modal genomics data'
        },
        'datasets': []
    }
    
    # Get list of datasets
    datasets = get_cohort_datasets(code, name)
    
    # Fetch metadata for each dataset
    for dataset_id in sorted(datasets):
        print(f"  Fetching {dataset_id}...")
        metadata = fetch_dataset_metadata(dataset_id)
        
        if metadata:
            dataset_entry = {}
            
            # Map fields to YAML structure
            if 'dataset_id' in metadata:
                dataset_entry['dataset_id'] = metadata['dataset_id']
            else:
                dataset_entry['dataset_id'] = dataset_id
                
            for key in ['download', 'samples', 'version', 'assembly', 'unit', 
                       'platform', 'gene_mapping', 'author', 'wrangling', 'input_data_format']:
                if key in metadata:
                    if key == 'samples':
                        dataset_entry[key] = int(metadata[key])
                    else:
                        dataset_entry[key] = metadata[key]
            
            # Handle data_type specially
            if 'data_type' in metadata:
                dataset_entry['data_type'] = metadata['data_type'].replace(' ', '_')
            
            # Add raw_data URL
            if 'raw_data' in metadata:
                dataset_entry['raw_data'] = metadata['raw_data']
            else:
                dataset_entry['raw_data'] = 'https://docs.gdc.cancer.gov/Data/Release_Notes/Data_Release_Notes/#data-release-400'
            
            config['datasets'].append(dataset_entry)
        
        # Be respectful with rate limiting
        time.sleep(1)
    
    return config


def main():
    """Main function to generate all config files."""
    output_dir = Path("/workspace/data/xenabrowser/configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip BRCA as it's already done
    cancer_types_to_process = [ct for ct in CANCER_TYPES if ct[0] != 'BRCA']
    
    # Test with just CESC first
    cancer_types_to_process = [ct for ct in cancer_types_to_process if ct[0] == 'CESC']
    
    for code, name in cancer_types_to_process:
        try:
            config = generate_config(code, name)
            
            # Write to YAML file
            output_file = output_dir / f"{code.lower()}.yaml"
            
            with open(output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, 
                         allow_unicode=True, width=1000)
            
            print(f"✓ Created {output_file}")
            
        except Exception as e:
            print(f"✗ Error processing {code}: {e}")
            continue
    
    print("\n✓ All configuration files generated!")


if __name__ == "__main__":
    main()
