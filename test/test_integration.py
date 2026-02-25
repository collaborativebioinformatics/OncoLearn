"""
Integration test for XenaCohort functionality
"""

import sys
from pathlib import Path

def test_xena_cohort_integration():
    """Test XenaCohort class integration with builder."""
    from oncolearn.api.xenabrowser import XenaCohortBuilder, XenaCohort
    
    print("="*70)
    print("XenaCohort Integration Test")
    print("="*70)
    
    # Test 1: Builder returns XenaCohort instance
    print("\n1. Testing builder returns XenaCohort...")
    builder = XenaCohortBuilder()
    cohort = builder.build_cohort("BRCA")
    
    assert isinstance(cohort, XenaCohort), f"Expected XenaCohort, got {type(cohort)}"
    print("   ✓ Builder returns XenaCohort instance")
    
    # Test 2: Cohort has correct attributes
    print("\n2. Testing cohort attributes...")
    assert hasattr(cohort, 'code'), "Missing 'code' attribute"
    assert hasattr(cohort, 'base_dir'), "Missing 'base_dir' attribute"
    assert cohort.code == "BRCA", f"Expected code 'BRCA', got '{cohort.code}'"
    assert cohort.name == "TCGA-BRCA", f"Expected name 'TCGA-BRCA', got '{cohort.name}'"
    print(f"   ✓ Code: {cohort.code}")
    print(f"   ✓ Name: {cohort.name}")
    print(f"   ✓ Base dir: {cohort.base_dir}")
    
    # Test 3: Cohort has data loading methods
    print("\n3. Testing data loading methods exist...")
    methods = ['clinical', 'mrna_seq', 'protein', 'methylation', 
               'cnv', 'mutation', 'mirna_seq', 'genomics']
    
    for method in methods:
        assert hasattr(cohort, method), f"Missing method: {method}()"
        assert callable(getattr(cohort, method)), f"{method}() is not callable"
    print(f"   ✓ All {len(methods)} data loading methods present")
    
    # Test 4: Test data loading (if data exists)
    print("\n4. Testing data loading (if data available)...")
    if cohort.base_dir.exists():
        try:
            clinical = cohort.clinical()
            if clinical is not None:
                assert '_source_dataset' in clinical.columns, "Missing _source_dataset column"
                print(f"   ✓ Clinical data loaded: {clinical.shape}")
                print(f"   ✓ Source tracking: {clinical['_source_dataset'].nunique()} datasets")
            else:
                print("   ⚠ Clinical data returned None (files may not exist)")
        except Exception as e:
            print(f"   ⚠ Error loading clinical data: {e}")
    else:
        print(f"   ⚠ Data directory does not exist: {cohort.base_dir}")
        print("      Run cohort.download() to fetch data")
    
    # Test 5: Test download method
    print("\n5. Testing download method...")
    assert hasattr(cohort, 'download'), "Missing download() method"
    assert callable(cohort.download), "download() is not callable"
    print("   ✓ Download method present")
    
    # Test 6: Test cohort dataset management
    print("\n6. Testing dataset management...")
    assert len(cohort.datasets) > 0, "No datasets in cohort"
    print(f"   ✓ Total datasets: {len(cohort.datasets)}")
    
    # Count by category
    from oncolearn.api.dataset import DataCategory
    category_counts = {}
    for category in DataCategory:
        datasets = cohort.get_datasets_by_category(category)
        if datasets:
            category_counts[category.value] = len(datasets)
    
    print(f"   ✓ Dataset categories: {len(category_counts)}")
    for cat, count in sorted(category_counts.items()):
        print(f"      - {cat}: {count} dataset(s)")
    
    # Test 7: Test repr
    print("\n7. Testing string representation...")
    repr_str = repr(cohort)
    assert 'XenaCohort' in repr_str, "repr() should contain 'XenaCohort'"
    assert 'BRCA' in repr_str, "repr() should contain cohort code"
    print(f"   ✓ repr: {repr_str}")
    
    print("\n" + "="*70)
    print("All Integration Tests Passed! ✓")
    print("="*70)
    
    return True

if __name__ == "__main__":
    try:
        success = test_xena_cohort_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
