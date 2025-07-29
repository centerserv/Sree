#!/usr/bin/env python3

import numpy as np
import pandas as pd
import traceback
import sys
from datetime import datetime

def test_heart_disease_dataset():
    print(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] Testing heart disease dataset...")
    try:
        # Load the actual dataset
        df = pd.read_csv('heart_disease_dataset_new.csv')
        
        print(f"✅ Dataset loaded successfully")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"🎯 Target column candidates: {[col for col in df.columns if 'target' in col.lower()]}")
        
        # Check for target column
        if 'target' in df.columns:
            target_col = 'target'
        else:
            # Look for common target column names
            possible_targets = [col for col in df.columns if any(word in col.lower() for word in ['target', 'class', 'label', 'outcome', 'result'])]
            if possible_targets:
                target_col = possible_targets[0]
                print(f"⚠️ Using '{target_col}' as target column")
            else:
                # Use last column as target
                target_col = df.columns[-1]
                print(f"⚠️ Using last column '{target_col}' as target")
        
        # Prepare features and target
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        print(f"✅ Features shape: {X.shape}")
        print(f"✅ Target shape: {y.shape}")
        print(f"✅ Target classes: {np.unique(y)}")
        print(f"✅ Target distribution: {np.bincount(y)}")
        
        # Check for any problematic values
        print(f"🔍 Checking data quality...")
        print(f"   - NaN values in X: {np.isnan(X).sum()}")
        print(f"   - Inf values in X: {np.isinf(X).sum()}")
        print(f"   - X min: {X.min()}, max: {X.max()}")
        print(f"   - X mean: {X.mean():.6f}, std: {X.std():.6f}")
        
        return True, X, y
    except Exception as e:
        print(f"❌ Heart disease dataset error: {e}")
        traceback.print_exc()
        return False, None, None

def test_sree_with_real_data(X, y):
    print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] Testing SREE with real heart disease data...")
    try:
        from unified_block_creation import run_unified_block_creation
        
        print("🚀 Starting SREE processing with real data...")
        print(f"📊 Processing {X.shape[0]} samples with {X.shape[1]} features")
        
        # Start the processing
        results = run_unified_block_creation(X, y, dataset_name="heart_disease_debug")
        
        print(f"✅ SREE processing completed successfully!")
        print(f"📊 Final Results:")
        print(f"   - Accuracy: {results.get('accuracy', 0.0):.3f}")
        print(f"   - Trust Score: {results.get('trust_score', 0.0):.3f}")
        print(f"   - Entropy: {results.get('entropy', 0.0):.3f}")
        print(f"   - Block Count: {results.get('block_count', 0)}")
        print(f"   - All OK: {results.get('all_ok', False)}")
        
        return True
    except Exception as e:
        print(f"❌ SREE processing with real data error: {e}")
        traceback.print_exc()
        return False

def test_dashboard_simulation():
    print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] Simulating dashboard workflow...")
    try:
        from dashboard import SREEDashboard
        
        # Load dataset
        df = pd.read_csv('heart_disease_dataset_new.csv')
        
        # Simulate dashboard workflow
        dashboard = SREEDashboard()
        
        # Find target column
        if 'target' in df.columns:
            target_col = 'target'
        else:
            target_col = df.columns[-1]
        
        # Prepare data like dashboard does
        feature_columns = [col for col in df.columns if col != target_col]
        X = df[feature_columns].values
        y = df[target_col].values
        
        print(f"🎯 Simulating dashboard analysis...")
        print(f"   - Dataset shape: {X.shape}")
        print(f"   - Target classes: {len(np.unique(y))}")
        
        # Run analysis like dashboard
        results = dashboard.run_sree_analysis(X, y)
        
        print(f"✅ Dashboard simulation completed!")
        print(f"📊 Results: {results}")
        
        return True
    except Exception as e:
        print(f"❌ Dashboard simulation error: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔧 SREE Heart Disease Debug Tool")
    print("=" * 50)
    
    # Test 1: Load and examine heart disease dataset
    success, X, y = test_heart_disease_dataset()
    if not success:
        print("❌ Heart disease dataset loading failed. Exiting.")
        sys.exit(1)
    
    # Test 2: Process with SREE directly
    if not test_sree_with_real_data(X, y):
        print("❌ SREE processing with real data failed.")
        # Don't exit, continue with dashboard test
    
    # Test 3: Simulate dashboard workflow
    if not test_dashboard_simulation():
        print("❌ Dashboard simulation failed.")
        sys.exit(1)
    
    print("\n🎉 Heart disease dataset tests completed!")

if __name__ == "__main__":
    main() 