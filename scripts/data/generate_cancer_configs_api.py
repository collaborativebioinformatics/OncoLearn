#!/usr/bin/env python3
"""
Script to generate YAML configuration files for TCGA cancer types from Xena Browser API.
"""

import time
from pathlib import Path
from typing import Dict, Optional

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

# Standard dataset patterns for GDC TCGA cohorts
DATASET_PATTERNS = [
    "allele_cnv_ascat2.tsv",
    "allele_cnv_ascat3.tsv",
    "segment_cnv_ascat-ngs.tsv",
    "masked_cnv_DNAcopy.tsv",
    "gene-level_absolute.tsv",
    "gene-level_ascat2.tsv",
    "gene-level_ascat3.tsv",
    "gene-level_ascat-ngs.tsv",
    "methylation27.tsv",
    "methylation450.tsv",
    "star_counts.tsv",
    "star_fpkm.tsv",
    "star_fpkm-uq.tsv",
    "star_tpm.tsv",
    "clinical.tsv",
    "survival.tsv",
    "protein.tsv",
    "somaticmutation_wxs.tsv",
    "mirna.tsv",
]


def fetch_json_metadata(dataset_id: str) -> Optional[Dict]:
    """Fetch JSON metadata file for a dataset from S3."""
    # Try to get the JSON metadata file
    json_url = f"https://gdc-hub.s3.us-east-1.amazonaws.com/download/{dataset_id}.json"
    
    try:
        response = requests.get(json_url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    
    return None


def get_sample_count(dataset_id: str, data_format: str) -> Optional[int]:
    """Get sample count by fetching a portion of the data file."""
    data_url = f"https://gdc-hub.s3.us-east-1.amazonaws.com/download/{dataset_id}.gz"
    
    try:
        # Fetch only the first few KB to get the header
        response = requests.get(data_url, timeout=10, stream=True, headers={'Range': 'bytes=0-8192'})
        
        if response.status_code in [200, 206]:
            import gzip
            
            # Decompress the data
            decompressed = gzip.decompress(response.content)
            lines = decompressed.decode('utf-8', errors='ignore').split('\n')
            
            if lines:
                header = lines[0]
                
                # For genomicMatrix format, count columns minus 1 (first column is row ID)
                if 'genomicMatrix' in data_format or 'clinicalMatrix' in data_format:
                    columns = header.split('\t')
                    return len(columns) - 1 if len(columns) > 1 else 0
                
                # For genomicSegment, count unique samples
                elif 'genomicSegment' in data_format:
                    samples = set()
                    for line in lines[1:min(100, len(lines))]:
                        if line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 4:
                                samples.add(parts[3])  # Sample ID is typically 4th column
                    
                    # If we found samples in first 100 lines, estimate total
                    # Otherwise we'll need to scan more, but let's return what we have
                    if samples:
                        return len(samples)
                
                # For mutationVector, count unique samples similarly
                elif 'mutationVector' in data_format:
                    samples = set()
                    for line in lines[1:min(100, len(lines))]:
                        if line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                samples.add(parts[0])  # Sample ID is typically first column
                    if samples:
                        return len(samples)
    
    except Exception:
        pass
    
    return None


def get_dataset_info(code: str, pattern: str) -> Optional[Dict]:
    """Get dataset information for a specific cancer type and pattern."""
    dataset_id = f"TCGA-{code}.{pattern}"
    
    # Fetch JSON metadata
    json_meta = fetch_json_metadata(dataset_id)
    
    if not json_meta:
        return None
    
    dataset_info = {
        'dataset_id': dataset_id,
        'download': f"https://gdc-hub.s3.us-east-1.amazonaws.com/download/{dataset_id}.gz",
    }
    
    # Determine data format first
    data_format = ''
    if 'type' in json_meta:
        data_format = json_meta['type']
    
    # Extract information from JSON metadata
    sample_count = None
    if 'sampleCount' in json_meta:
        sample_count = json_meta['sampleCount']
    
    # Try to get sample count from data file
    if sample_count is None:
        sample_count = get_sample_count(dataset_id, data_format)
    
    if sample_count is not None:
        dataset_info['samples'] = sample_count
    
    if 'version' in json_meta:
        dataset_info['version'] = json_meta['version']
    
    if 'dataSubType' in json_meta:
        data_type = json_meta['dataSubType']
        # Map data subtypes to standardized names
        type_mapping = {
            'copy number': 'copy_number',
            'copy number (gene-level)': 'copy_number_gene_level',
            'DNA methylation': 'DNA_methylation',
            'gene expression RNAseq': 'gene_expression_RNAseq',
            'phenotype': 'phenotype',
            'protein expression': 'protein_expression',
            'somatic mutation (SNPs and small INDELs)': 'somatic_mutation',
            'stem loop expression': 'stem_loop_expression',
        }
        dataset_info['data_type'] = type_mapping.get(data_type, data_type.replace(' ', '_'))
    
    if 'assembly' in json_meta:
        dataset_info['assembly'] = json_meta['assembly']
    
    if 'unit' in json_meta:
        dataset_info['unit'] = json_meta['unit']
    
    if 'platform' in json_meta:
        dataset_info['platform'] = json_meta['platform']
    
    if 'probeMap' in json_meta:
        dataset_info['gene_mapping'] = json_meta['probeMap']
    
    # Standard fields for GDC data
    dataset_info['author'] = "Genomic Data Commons"
    
    if 'url' in json_meta:
        # The URL field contains raw data URLs
        urls = json_meta['url'].split(', ')
        if urls:
            dataset_info['raw_data'] = urls[0]
    else:
        dataset_info['raw_data'] = "https://docs.gdc.cancer.gov/Data/Release_Notes/Data_Release_Notes/#data-release-400"
    
    if 'wrangling' in json_meta:
        dataset_info['wrangling'] = json_meta['wrangling']
    elif 'wrangling_procedure' in json_meta:
        dataset_info['wrangling'] = json_meta['wrangling_procedure']
    else:
        # Add default wrangling descriptions based on data type
        if 'copy_number' in dataset_info.get('data_type', ''):
            if 'gene-level' in pattern:
                dataset_info['wrangling'] = "Loaded data directly into Xena"
            else:
                dataset_info['wrangling'] = "Chromosome location and segment mean data are presented."
        elif 'methylation' in pattern:
            dataset_info['wrangling'] = "Beta_value from the same sample but from different vials/portions/analytes/aliquotes is averaged; beta_value from different samples is combined into genomicMatrix."
        elif 'star_' in pattern:
            if 'counts' in pattern:
                dataset_info['wrangling'] = "Data from the same sample but from different vials/portions/analytes/aliquotes is averaged; all data is then log2(x+1) transformed."
            else:
                dataset_info['wrangling'] = "Data from the same sample but from different vials/portions/analytes/aliquotes is averaged; all data is then log2(x+1) transformed."
        elif 'protein' in pattern:
            dataset_info['wrangling'] = "value from different samples are combined into genomicMatrix"
        elif 'somaticmutation' in pattern:
            dataset_info['wrangling'] = "Hugo_Symbol, Chromosome, Start_Position, End_Position, Reference_Allele, Tumor_Seq_Allele2, HGVSp_Short and Consequence data are renamed accordingly and presented; dna_vaf data is added and is calculated by t_alt_count / t_depth."
        elif 'mirna' in pattern:
            dataset_info['wrangling'] = "RPM Data from the same sample but from different vials/portions/analytes/aliquotes is averaged; all data is then log2(x+1) transformed."
    
    if 'type' in json_meta:
        format_type = json_meta['type']
        format_mapping = {
            'genomicMatrix': 'ROWs (identifiers) x COLUMNs (samples) (i.e. genomicMatrix)',
            'clinicalMatrix': 'ROWs (samples) x COLUMNs (identifiers) (i.e. clinicalMatrix)',
            'genomicSegment': 'Genomic Segment (i.e. genomicSegment)',
            'mutationVector': 'Variant by Position (i.e. mutationVector)',
        }
        dataset_info['input_data_format'] = format_mapping.get(format_type, format_type)
    
    return dataset_info


def generate_config(code: str, name: str) -> Dict:
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
    
    # Try each standard dataset pattern
    for pattern in DATASET_PATTERNS:
        print(f"  Checking {pattern}...", end=' ')
        dataset_info = get_dataset_info(code, pattern)
        
        if dataset_info:
            print(f"✓ (n={dataset_info.get('samples', '?')})")
            config['datasets'].append(dataset_info)
        else:
            print("✗")
        
        # Rate limiting
        time.sleep(0.5)
    
    return config


def main():
    """Main function to generate all config files."""
    output_dir = Path("/workspace/data/xenabrowser/configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip BRCA as it's already done
    cancer_types_to_process = [ct for ct in CANCER_TYPES if ct[0] != 'BRCA']
    
    for code, name in cancer_types_to_process:
        try:
            config = generate_config(code, name)
            
            if config['datasets']:
                # Write to YAML file
                output_file = output_dir / f"{code.lower()}.yaml"
                
                with open(output_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False, 
                             allow_unicode=True, width=1000)
                
                print(f"✓ Created {output_file} with {len(config['datasets'])} datasets")
            else:
                print(f"✗ No datasets found for {code}")
            
        except Exception as e:
            print(f"✗ Error processing {code}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n✓ All configuration files generated!")


if __name__ == "__main__":
    main()
