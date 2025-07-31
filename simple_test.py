#!/usr/bin/env python3
"""
Simple Test to identify the problem
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import logging

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import setup_logging

def test_simple_dataset():
    """Test with a very simple dataset."""
    print("🚀 Simple Test")
    print("=" * 30)
    
    # Create a very small dataset
    print("📊 Creating 100 row dataset...")
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42,
        n_classes=2
    )
    
    print(f"✅ Dataset created: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"📈 Target distribution: {np.bincount(y)}")
    
    # Test basic operations
    print(f"\n⏱️  Testing basic operations...")
    start_time = time.time()
    
    try:
        # Test 1: Basic numpy operations
        print("   Testing numpy operations...")
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
        print(f"   ✅ Numpy scaling: {X_scaled.shape}")
        
        # Test 2: Import pattern layer
        print("   Testing pattern layer import...")
        from layers.pattern import PatternValidator
        print("   ✅ Pattern layer imported")
        
        # Test 3: Create pattern validator
        print("   Testing pattern validator creation...")
        pattern_validator = PatternValidator()
        print("   ✅ Pattern validator created")
        
        # Test 4: Try to fit (this is where it fails)
        print("   Testing pattern fitting...")
        pattern_validator.fit(X_scaled, y)
        print("   ✅ Pattern fitting successful")
        
        end_time = time.time()
        print(f"\n✅ SUCCESS: All tests passed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    test_simple_dataset()
    
    print("\n🎯 Test completed!") 