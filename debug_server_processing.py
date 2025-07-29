#!/usr/bin/env python3

import numpy as np
import pandas as pd
import traceback
import sys
from datetime import datetime

def test_basic_imports():
    print(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] Testing basic imports...")
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} imported successfully")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__} imported successfully")
        
        from sklearn.model_selection import train_test_split
        print("✅ train_test_split imported successfully")
        
        from sklearn.preprocessing import StandardScaler
        print("✅ StandardScaler imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_sree_imports():
    print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] Testing SREE imports...")
    try:
        from unified_block_creation import run_unified_block_creation
        print("✅ unified_block_creation imported successfully")
        
        from layers.pattern import PatternValidator
        print("✅ PatternValidator imported successfully")
        
        from layers.presence import PresenceValidator
        print("✅ PresenceValidator imported successfully")
        
        from layers.permanence import PermanenceValidator
        print("✅ PermanenceValidator imported successfully")
        
        from layers.logic import LogicValidator
        print("✅ LogicValidator imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ SREE import error: {e}")
        traceback.print_exc()
        return False

def test_data_processing():
    print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] Testing data processing...")
    try:
        # Create small test dataset
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        print(f"✅ Test data created: X shape {X.shape}, y shape {y.shape}")
        print(f"✅ X stats: min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}")
        print(f"✅ y stats: unique values {np.unique(y)}, counts {np.bincount(y)}")
        
        # Test train_test_split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"✅ Train/test split successful: train {X_train.shape}, test {X_test.shape}")
        
        # Test StandardScaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"✅ Scaling successful: scaled train {X_train_scaled.shape}")
        
        return True, X, y
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        traceback.print_exc()
        return False, None, None

def test_sree_processing(X, y):
    print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] Testing SREE processing...")
    try:
        from unified_block_creation import run_unified_block_creation
        
        print("🚀 Starting SREE processing with debug logging...")
        results = run_unified_block_creation(X, y, dataset_name="debug_test")
        
        print(f"✅ SREE processing completed successfully!")
        print(f"📊 Results: {results}")
        
        return True
    except Exception as e:
        print(f"❌ SREE processing error: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔧 SREE Server Debug Tool")
    print("=" * 50)
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("❌ Basic imports failed. Exiting.")
        sys.exit(1)
    
    # Test 2: SREE imports
    if not test_sree_imports():
        print("❌ SREE imports failed. Exiting.")
        sys.exit(1)
    
    # Test 3: Data processing
    success, X, y = test_data_processing()
    if not success:
        print("❌ Data processing failed. Exiting.")
        sys.exit(1)
    
    # Test 4: SREE processing
    if not test_sree_processing(X, y):
        print("❌ SREE processing failed. Exiting.")
        sys.exit(1)
    
    print("\n🎉 All tests passed successfully!")
    print("✅ The server should be able to process datasets correctly.")

if __name__ == "__main__":
    main() 