"""
SREE Phase 1 Demo - Setup Test
Test script to verify environment setup and basic functionality.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        
        import pandas as pd
        print("✓ Pandas imported successfully")
        
        import sklearn
        print("✓ Scikit-learn imported successfully")
        
        import matplotlib
        print("✓ Matplotlib imported successfully")
        
        import seaborn
        print("✓ Seaborn imported successfully")
        
        import joblib
        print("✓ Joblib imported successfully")
        
        
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        assert False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config import setup_logging, get_config
        
        # Test logging setup
        logger = setup_logging(level="INFO")
        logger.info("Test log message")
        print("✓ Logging configuration working")
        
        # Test config loading
        config = get_config()
        print(f"✓ Configuration loaded: {len(config)} sections")
        
        
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        assert False

def test_data_loader():
    """Test data loader functionality."""
    print("\nTesting data loader...")
    
    try:
        from data_loader import DataLoader
        from config import setup_logging
        
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Test synthetic data creation (fastest)
        X, y = loader.create_synthetic(n_samples=100)
        print(f"✓ Synthetic data created: X.shape={X.shape}, y.shape={y.shape}")
        
        # Test preprocessing
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        print(f"✓ Data preprocessing: train={X_train.shape}, test={X_test.shape}")
        
        # Test dataset info
        info = loader.get_dataset_info(X, y)
        print(f"✓ Dataset info: {info['n_samples']} samples, {info['n_classes']} classes")
        
        
        
    except Exception as e:
        print(f"✗ Data loader error: {e}")
        assert False

def test_validator_interface():
    """Test base validator interface."""
    print("\nTesting validator interface...")
    
    try:
        from layers.base import Validator
        import numpy as np
        
        # Test that we can't instantiate abstract class
        try:
            validator = Validator()
            print("✗ Should not be able to instantiate abstract Validator")
            assert False
        except TypeError:
            print("✓ Abstract Validator class properly protected")
        
        # Test metadata method
        class TestValidator(Validator):
            def validate(self, data, labels=None):
                return np.ones(len(data))
        
        test_val = TestValidator("test")
        metadata = test_val.get_metadata()
        print(f"✓ Validator metadata: {metadata}")
        
        
        
    except Exception as e:
        print(f"✗ Validator interface error: {e}")
        assert False

def test_directory_structure():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = ["data", "models", "layers", "loop", "logs", "plots", "tests"]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ Directory {dir_name} exists")
        else:
            print(f"✗ Directory {dir_name} missing")
            assert False
    
    

def main():
    """Run all setup tests."""
    print("SREE Phase 1 Demo - Environment Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Loader", test_data_loader),
        ("Validator Interface", test_validator_interface),
        ("Directory Structure", test_directory_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Environment setup is complete.")
        print("\nNext steps:")
        print("1. Implement Pattern Layer (MLP classifier)")
        print("2. Implement Presence Layer (entropy minimization)")
        print("3. Implement Permanence Layer (hash-based logging)")
        print("4. Implement Logic Layer (consistency validation)")
        print("5. Implement Trust Update Loop")
        
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        assert False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 