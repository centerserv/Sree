#!/usr/bin/env python3
"""
Test Heart Large Performance
Test the performance improvements for heart_large.csv (30,000 rows)
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from unified_block_creation import run_unified_block_creation
from config import setup_logging

def load_heart_large_dataset():
    """Load the heart_large.csv dataset."""
    print("📊 Loading heart_large.csv dataset...")
    
    try:
        # Load the dataset
        df = pd.read_csv('heart_large.csv')
        
        # Separate features and target
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        print(f"✅ Dataset loaded: {X.shape[0]:,} rows, {X.shape[1]} features")
        print(f"📈 Target distribution: {np.bincount(y)}")
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None, None

def test_heart_large_performance():
    """Test performance with heart_large.csv dataset."""
    print("🚀 Heart Large Performance Test")
    print("=" * 50)
    
    # Load dataset
    X, y = load_heart_large_dataset()
    
    if X is None or y is None:
        print("❌ Failed to load dataset")
        return
    
    print(f"\n📊 Dataset Info:")
    print(f"   Rows: {X.shape[0]:,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target classes: {len(np.unique(y))}")
    print(f"   Memory usage: {X.nbytes / 1024 / 1024:.2f} MB")
    
    # Test performance
    print(f"\n⏱️  Starting analysis...")
    start_time = time.time()
    
    try:
        results = run_unified_block_creation(
            X=X,
            y=y,
            accuracy_threshold=0.85,  # Relaxed for large datasets
            trust_threshold=0.75,    # Relaxed for large datasets
            entropy_threshold=2.5,   # Relaxed for large datasets
            max_blocks=3,            # Limited for speed
            required_consecutive_ok=1, # Stop after first good block
            dataset_name="heart_large",
            use_dashboard_config=False
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if results:
            accuracy = results.get('accuracy', 0.0)
            trust = results.get('trust_score', 0.0)
            entropy = results.get('entropy', 0.0)
            block_count = results.get('block_count', 0)
            stop_reason = results.get('stop_reason', 'Unknown')
            
            print(f"\n📊 RESULTS FOR HEART_LARGE.CSV:")
            print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
            print(f"📈 Final Accuracy: {accuracy:.3f}")
            print(f"🔒 Final Trust: {trust:.3f}")
            print(f"📊 Final Entropy: {entropy:.3f}")
            print(f"🏗️  Blocks Created: {block_count}")
            print(f"🛑 Stop Reason: {stop_reason}")
            
            # Compare with user's result (30m 40s = 1840s)
            user_time = 30 * 60 + 40  # 30 minutes 40 seconds
            improvement = user_time / execution_time
            
            print(f"\n📈 PERFORMANCE COMPARISON:")
            print(f"   User's time: {user_time:.0f} seconds (30m 40s)")
            print(f"   New time: {execution_time:.2f} seconds")
            print(f"   Improvement: {improvement:.1f}x faster!")
            
            # Performance assessment
            if execution_time < 60:  # Less than 1 minute
                print(f"🎉 EXCELLENT: {execution_time:.2f}s - Massive improvement!")
            elif execution_time < 300:  # Less than 5 minutes
                print(f"✅ GOOD: {execution_time:.2f}s - Significant improvement!")
            elif execution_time < 600:  # Less than 10 minutes
                print(f"⚠️  ACCEPTABLE: {execution_time:.2f}s - Moderate improvement")
            else:
                print(f"❌ SLOW: {execution_time:.2f}s - Still needs optimization")
            
            # Show configuration used
            print(f"\n⚙️  Configuration Applied:")
            print(f"   Dataset size: {X.shape[0]:,} rows")
            print(f"   Should use: LARGE_DATASET_CONFIG")
            print(f"   Expected iterations: 5")
            print(f"   Expected max blocks: 3")
            
        else:
            print(f"❌ Failed to get results")
            
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"❌ Error: {str(e)}")
        print(f"⏱️  Time before error: {execution_time:.2f} seconds")

def test_heart_large_with_dashboard():
    """Test heart_large.csv using dashboard configuration."""
    print("\n🖥️  Testing with Dashboard Configuration")
    print("=" * 50)
    
    # Load dataset
    X, y = load_heart_large_dataset()
    
    if X is None or y is None:
        print("❌ Failed to load dataset")
        return
    
    # Test performance with dashboard config
    print(f"\n⏱️  Starting analysis with dashboard config...")
    start_time = time.time()
    
    try:
        results = run_unified_block_creation(
            X=X,
            y=y,
            accuracy_threshold=0.85,
            trust_threshold=0.75,
            entropy_threshold=2.5,
            max_blocks=8,  # Dashboard config uses 8 blocks
            required_consecutive_ok=2, # Dashboard config uses 2
            dataset_name="heart_large_dashboard",
            use_dashboard_config=True  # Use dashboard config
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if results:
            accuracy = results.get('accuracy', 0.0)
            trust = results.get('trust_score', 0.0)
            entropy = results.get('entropy', 0.0)
            block_count = results.get('block_count', 0)
            
            print(f"\n📊 RESULTS WITH DASHBOARD CONFIG:")
            print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
            print(f"📈 Final Accuracy: {accuracy:.3f}")
            print(f"🔒 Final Trust: {trust:.3f}")
            print(f"📊 Final Entropy: {entropy:.3f}")
            print(f"🏗️  Blocks Created: {block_count}")
            
            # Compare with user's result
            user_time = 30 * 60 + 40  # 30 minutes 40 seconds
            improvement = user_time / execution_time
            
            print(f"\n📈 PERFORMANCE COMPARISON (Dashboard Config):")
            print(f"   User's time: {user_time:.0f} seconds (30m 40s)")
            print(f"   New time: {execution_time:.2f} seconds")
            print(f"   Improvement: {improvement:.1f}x faster!")
            
        else:
            print(f"❌ Failed to get results with dashboard config")
            
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"❌ Error with dashboard config: {str(e)}")
        print(f"⏱️  Time before error: {execution_time:.2f} seconds")

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    print("🚀 Heart Large Performance Test")
    print("Testing the new LARGE_DATASET_CONFIG optimization")
    print("=" * 60)
    
    # Test with large dataset config
    test_heart_large_performance()
    
    # Test with dashboard config for comparison
    test_heart_large_with_dashboard()
    
    print("\n🎯 Test completed!") 