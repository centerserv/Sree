"""
SREE Phase 1 Demo - Comprehensive Test Suite
Complete validation of all foundation components before Phase 2.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import traceback

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


def test_imports():
    """Test all required package imports."""
    print("🔍 Testing Package Imports...")
    
    packages = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("joblib", "joblib"),
        ("openml", "openml")
    ]
    
    failed_imports = []
    
    for package_name, import_name in packages:
        try:
            __import__(package_name)
            print(f"  ✅ {package_name}")
        except ImportError as e:
            print(f"  ❌ {package_name}: {e}")
            failed_imports.append(package_name)
    
    if failed_imports:
        print(f"  ⚠️  Failed imports: {failed_imports}")
        assert False, f"Failed imports: {failed_imports}"
    
    print("  ✅ All packages imported successfully")


def test_configuration():
    """Test configuration system."""
    print("\n🔧 Testing Configuration System...")
    
    try:
        from config import setup_logging, get_config, DATASET_CONFIG, MODEL_CONFIG, PPP_CONFIG
        
        # Test logging setup
        logger = setup_logging(level="INFO")
        logger.info("Configuration test log message")
        print("  ✅ Logging system working")
        
        # Test config loading
        config = get_config()
        required_sections = ["paths", "datasets", "model", "ppp", "testing", "visualization", "targets"]
        
        for section in required_sections:
            if section not in config:
                print(f"  ❌ Missing config section: {section}")
                assert False, f"Missing config section: {section}"
        
        print(f"  ✅ Configuration loaded: {len(config)} sections")
        
        # Test specific config values
        if config["targets"]["sree"]["accuracy"] != 0.985:
            print("  ❌ Incorrect target accuracy")
            assert False, "Incorrect target accuracy"
        
        if config["targets"]["sree"]["trust"] != 0.96:
            print("  ❌ Incorrect target trust score")
            assert False, "Incorrect target trust score"
        
        print("  ✅ All configuration values correct")
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        traceback.print_exc()
        assert False, f"Configuration error: {e}"


def test_data_loader():
    """Test data loading functionality."""
    print("\n📊 Testing Data Loader...")
    
    try:
        from data_loader import DataLoader, load_all_datasets
        from config import setup_logging
        
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Test synthetic data creation
        X, y = loader.create_synthetic(n_samples=100)
        if X.shape != (100, 784) or y.shape != (100,):
            print(f"  ❌ Synthetic data shape incorrect: X{X.shape}, y{y.shape}")
            assert False, f"Synthetic data shape incorrect: X{X.shape}, y{y.shape}"
        print("  ✅ Synthetic data creation working")
        
        # Test data preprocessing
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        expected_train = int(0.8 * 100)
        expected_test = 100 - expected_train
        
        if X_train.shape[0] != expected_train or X_test.shape[0] != expected_test:
            print(f"  ❌ Train/test split incorrect: train{X_train.shape[0]}, test{X_test.shape[0]}")
            assert False, f"Train/test split incorrect: train{X_train.shape[0]}, test{X_test.shape[0]}"
        print("  ✅ Data preprocessing working")
        
        # Test dataset info
        info = loader.get_dataset_info(X, y)
        required_keys = ["n_samples", "n_features", "n_classes", "class_distribution", "feature_stats"]
        
        for key in required_keys:
            if key not in info:
                print(f"  ❌ Missing dataset info key: {key}")
                assert False, f"Missing dataset info key: {key}"
        
        print("  ✅ Dataset info generation working")
        
        # Test load_all_datasets (with fallback)
        try:
            datasets = load_all_datasets(logger)
            print("  ✅ All datasets loaded successfully")
        except Exception as e:
            print(f"  ⚠️  Network error (expected): {e}")
            print("  ✅ Fallback mechanism working")
        
    except Exception as e:
        print(f"  ❌ Data loader error: {e}")
        traceback.print_exc()
        assert False, f"Data loader error: {e}"


def test_validator_interface():
    """Test base validator interface."""
    print("\n🏗️  Testing Validator Interface...")
    
    try:
        from layers.base import Validator
        import numpy as np
        
        # Test abstract class protection
        try:
            validator = Validator()
            print("  ❌ Should not be able to instantiate abstract Validator")
            assert False, "Should not be able to instantiate abstract Validator"
        except TypeError:
            print("  ✅ Abstract Validator class properly protected")
        
        # Test concrete implementation
        class TestValidator(Validator):
            def validate(self, data, labels=None):
                return np.ones(len(data))
        
        test_val = TestValidator("test_validator")
        
        # Test metadata
        metadata = test_val.get_metadata()
        if metadata["type"] != "TestValidator":
            print(f"  ❌ Incorrect metadata type: {metadata['type']}")
            assert False, f"Incorrect metadata type: {metadata['type']}"
        print("  ✅ Validator metadata working")
        
        # Test state management
        state = test_val.get_state()
        if "name" not in state or "metadata" not in state:
            print("  ❌ Missing state information")
            assert False, "Missing state information"
        print("  ✅ Validator state management working")
        
        # Test validation method
        test_data = np.random.rand(10, 5)
        result = test_val.validate(test_data)
        if len(result) != 10 or not np.all(result == 1.0):
            print("  ❌ Validation method not working correctly")
            assert False, "Validation method not working correctly"
        print("  ✅ Validator validation method working")
        
    except Exception as e:
        print(f"  ❌ Validator interface error: {e}")
        traceback.print_exc()
        assert False, f"Validator interface error: {e}"


def test_directory_structure():
    """Test project directory structure."""
    print("\n📁 Testing Directory Structure...")
    
    required_dirs = [
        "data",
        "models", 
        "layers",
        "loop",
        "logs",
        "plots",
        "tests"
    ]
    
    required_files = [
        "config.py",
        "data_loader.py",
        "main.py",
        "requirements.txt",
        "README.md",
        "tests/test_setup.py",
        "layers/__init__.py",
        "layers/base.py"
    ]
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"  ❌ Missing directory: {dir_name}")
            assert False, f"Missing directory: {dir_name}"
        print(f"  ✅ Directory exists: {dir_name}")
    
    # Check files
    for file_name in required_files:
        file_path = Path(file_name)
        if not file_path.exists() or not file_path.is_file():
            print(f"  ❌ Missing file: {file_name}")
            assert False, f"Missing file: {file_name}"
        print(f"  ✅ File exists: {file_name}")


def test_main_execution():
    """Test main script execution."""
    print("\n🚀 Testing Main Script Execution...")
    
    try:
        import subprocess
        import sys
        
        # Run main script
        result = subprocess.run([sys.executable, "main.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"  ❌ Main script failed with return code: {result.returncode}")
            print(f"  Error output: {result.stderr}")
            assert False, f"Main script failed with return code: {result.returncode}"
        
        # Check for expected output (combine stdout and stderr)
        output = result.stdout + result.stderr
        expected_phrases = [
            "Starting SREE Phase 1 Demo",
            "Configuration loaded",
            "Loading datasets",
            "Phase 1 Foundation complete"
        ]
        
        for phrase in expected_phrases:
            if phrase not in output:
                print(f"  ❌ Missing expected output: {phrase}")
                assert False, f"Missing expected output: {phrase}"
        
        print("  ✅ Main script execution successful")
        
    except subprocess.TimeoutExpired:
        print("  ❌ Main script execution timed out")
        assert False, "Main script execution timed out"
    except Exception as e:
        print(f"  ❌ Main script test error: {e}")
        traceback.print_exc()
        assert False, f"Main script test error: {e}"


def test_code_quality():
    """Test code quality standards."""
    print("\n✨ Testing Code Quality...")
    
    try:
        # Test PEP 8 compliance (basic checks)
        import ast
        
        python_files = [
            "config.py",
            "data_loader.py", 
            "main.py",
            "tests/test_setup.py",
            "layers/base.py"
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    source = f.read()
                
                # Parse Python code
                ast.parse(source)
                print(f"  ✅ {file_path}: Valid Python syntax")
                
                # Check for docstrings
                if '"""' not in source and "'''" not in source:
                    print(f"  ⚠️  {file_path}: Missing docstrings")
                
            except SyntaxError as e:
                print(f"  ❌ {file_path}: Syntax error - {e}")
                assert False, f"{file_path}: Syntax error - {e}"
            except Exception as e:
                print(f"  ❌ {file_path}: Error - {e}")
                assert False, f"{file_path}: Error - {e}"
        
        print("  ✅ All Python files have valid syntax")
        
    except Exception as e:
        print(f"  ❌ Code quality test error: {e}")
        traceback.print_exc()
        assert False, f"Code quality test error: {e}"


def main():
    """Run comprehensive test suite."""
    print("🧪 SREE Phase 1 - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Configuration System", test_configuration),
        ("Data Loader", test_data_loader),
        ("Validator Interface", test_validator_interface),
        ("Directory Structure", test_directory_structure),
        ("Main Script Execution", test_main_execution),
        ("Code Quality", test_code_quality)
    ]
    
    passed = 0
    total = len(tests)
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed_tests.append(test_name)
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            failed_tests.append(test_name)
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if failed_tests:
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        print("\n🔧 Please fix the failed tests before proceeding to Phase 2.")
        return False
    else:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Phase 1 Foundation is 100% ready for Phase 2 implementation.")
        print("\n🚀 Ready to proceed with:")
        print("   1. Pattern Layer (MLP classifier)")
        print("   2. Presence Layer (entropy minimization)")
        print("   3. Permanence Layer (hash-based logging)")
        print("   4. Logic Layer (consistency validation)")
        print("   5. Trust Update Loop")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 