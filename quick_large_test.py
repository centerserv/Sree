#!/usr/bin/env python3
"""
Quick Large Dataset Test
Test if the metrics are no longer zero
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

from unified_block_creation import run_unified_block_creation
from config import setup_logging

def test_small_large_dataset():
    """Test with a smaller large dataset to verify metrics."""
    print("🚀 Quick Large Dataset Test")
    print("=" * 40)
    
    # Create a 5k dataset (large enough to trigger LARGE_DATASET_CONFIG)
    print("📊 Creating 5,000 row dataset...")
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=42,
        n_classes=2
    )
    
    print(f"✅ Dataset created: {X.shape[0]:,} rows, {X.shape[1]} features")
    
    # Test performance
    print(f"\n⏱️  Starting analysis...")
    start_time = time.time()
    
    try:
        results = run_unified_block_creation(
            X=X,
            y=y,
            accuracy_threshold=0.85,
            trust_threshold=0.75,
            entropy_threshold=2.5,
            max_blocks=5,
            required_consecutive_ok=2,
            dataset_name="quick_large_test",
            use_dashboard_config=False
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if results:
            accuracy = results.get('accuracy', 0.0)
            trust = results.get('trust_score', 0.0)
            entropy = results.get('entropy', 0.0)
            block_count = results.get('block_count', 0)
            
            print(f"\n📊 RESULTS:")
            print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
            print(f"📈 Final Accuracy: {accuracy:.3f}")
            print(f"🔒 Final Trust: {trust:.3f}")
            print(f"📊 Final Entropy: {entropy:.3f}")
            print(f"🏗️  Blocks Created: {block_count}")
            
            # Check if metrics are not zero
            if accuracy > 0 and trust > 0 and entropy > 0:
                print(f"✅ SUCCESS: Metrics are no longer zero!")
                print(f"🎉 Configuration fix worked!")
            else:
                print(f"❌ ERROR: Metrics are still zero!")
                print(f"   Accuracy: {accuracy}")
                print(f"   Trust: {trust}")
                print(f"   Entropy: {entropy}")
                
        else:
            print(f"❌ Failed to get results")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    test_small_large_dataset()
    
    print("\n🎯 Test completed!") 