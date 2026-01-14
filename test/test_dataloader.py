"""
Test the updated data loader with genetic data integration.
"""

import sys
from pathlib import Path

from oncolearn.utils.data_loader import MedicalImageDataset

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_genetic_data_loader():
    """Test loading data with genetic features."""
    print("="*60)
    print("Testing Genetic Data Integration")
    print("="*60)

    # Test with actual data paths
    data_dir = "/workspace/data/TCIA_BRCA/TCIA_TCGA-BRCA_09-16-2015_part1_of_4/TCGA-BRCA"
    genetic_data_dir = "/workspace/data/processed"

    print(f"\nData directory: {data_dir}")
    print(f"Genetic data directory: {genetic_data_dir}")

    # Check if directories exist
    import os
    if not os.path.exists(data_dir):
        print(f"Warning: Image directory not found: {data_dir}")
        print("Skipping test...")
        return

    if not os.path.exists(genetic_data_dir):
        print(f"Warning: Genetic data directory not found: {genetic_data_dir}")
        print("Skipping test...")
        return

    try:
        # Create dataset
        print("\nCreating dataset with BRCA data only...")
        dataset = MedicalImageDataset(
            data_dir=data_dir,
            genetic_data_dir=genetic_data_dir,
            image_size=(512, 512),
            use_genetic_data=True,
            max_genes=500,  # Use top 500 genes for testing
            extension="*.dcm",  # DICOM files
            cancer_type="BRCA",  # Only load BRCA data
        )

        print("\n[OK] Dataset created successfully!")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Num classes: {dataset.get_num_classes()}")
        print(f"  Num genes: {dataset.get_num_genes()}")

        # Test getting a sample
        if len(dataset) > 0:
            print("\nTesting sample retrieval...")
            sample = dataset[0]

            print("[OK] Sample retrieved successfully!")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Label: {sample['label'].item()}")
            print(f"  Patient ID: {sample['patient_id']}")

            if 'genetic' in sample:
                print(f"  Genetic features shape: {sample['genetic'].shape}")
                print(
                    f"  Genetic features mean: {sample['genetic'].mean():.4f}")
                print(f"  Genetic features std: {sample['genetic'].std():.4f}")
            else:
                print("  Warning: No genetic features in sample")

        print("\n" + "="*60)
        print("[OK] ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(test_genetic_data_loader())
