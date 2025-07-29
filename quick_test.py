#!/usr/bin/env python3
"""
SREE Quick Test Suite
Fast validation for deployment - runs in under 30 seconds
"""

import sys
import time
import subprocess
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def run_quick_imports():
    """Test critical imports only."""
    print("🔍 Testing Critical Imports...")
    
    critical_packages = [
        "numpy", "pandas", "sklearn", "matplotlib", "joblib"
    ]
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            return False
    
    return True

def run_quick_config():
    """Test configuration loading."""
    print("\n🔧 Testing Configuration...")
    
    try:
        from config import get_config
        config = get_config()
        
        required_sections = ["paths", "datasets", "model", "ppp"]
        for section in required_sections:
            if section not in config:
                print(f"  ❌ Missing: {section}")
                return False
        
        print("  ✅ Configuration loaded")
        return True
        
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False

def run_quick_data():
    """Test data loading with minimal samples."""
    print("\n📊 Testing Data Loading...")
    
    try:
        from data_loader import DataLoader
        from config import setup_logging
        
        logger = setup_logging(level="WARNING")  # Reduce logging
        loader = DataLoader(logger)
        
        # Test with minimal data
        X, y = loader.create_synthetic(n_samples=50)  # Very small sample
        
        if X.shape[0] == 50 and len(y) == 50:
            print("  ✅ Data loading working")
            return True
        else:
            print("  ❌ Data shape mismatch")
            return False
            
    except Exception as e:
        print(f"  ❌ Data error: {e}")
        return False

def run_quick_layers():
    """Test layer imports and basic functionality."""
    print("\n🏗️  Testing Core Layers...")
    
    try:
        from layers.pattern import PatternValidator
        from layers.presence import PresenceValidator
        from layers.permanence import PermanenceValidator
        from layers.logic import LogicValidator
        
        # Just test instantiation
        pattern = PatternValidator()
        presence = PresenceValidator()
        permanence = PermanenceValidator()
        logic = LogicValidator()
        
        print("  ✅ All layers imported")
        return True
        
    except Exception as e:
        print(f"  ❌ Layer error: {e}")
        return False

def run_quick_dashboard():
    """Test dashboard import."""
    print("\n📊 Testing Dashboard...")
    
    try:
        from dashboard import SREEDashboard
        print("  ✅ Dashboard imported")
        return True
        
    except Exception as e:
        print(f"  ❌ Dashboard error: {e}")
        return False

def run_unit_tests():
    """Run only the fastest unit tests."""
    print("\n🧪 Running Fast Unit Tests...")
    
    # Run only the fastest unit tests
    test_files = [
        "tests/test_setup.py",
        "tests/test_pattern_layer.py::test_pattern_validator_creation",
        "tests/test_presence_layer.py::test_presence_validator_creation"
    ]
    
    success_count = 0
    total_count = len(test_files)
    
    for test_file in test_files:
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_file, 
                "-v", "--tb=no", "--quiet"
            ], capture_output=True, timeout=10)
            
            if result.returncode == 0:
                print(f"  ✅ {test_file}")
                success_count += 1
            else:
                print(f"  ❌ {test_file}")
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {test_file} (timeout)")
        except Exception as e:
            print(f"  ❌ {test_file}: {e}")
    
    return success_count == total_count

def main():
    """Run quick test suite."""
    print("🚀 SREE Quick Test Suite")
    print("=" * 50)
    print("Fast validation for deployment (target: <30 seconds)")
    print()
    
    start_time = time.time()
    
    # Run quick tests
    tests = [
        ("Imports", run_quick_imports),
        ("Configuration", run_quick_config),
        ("Data Loading", run_quick_data),
        ("Core Layers", run_quick_layers),
        ("Dashboard", run_quick_dashboard),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
            break  # Stop on first failure
    
    # Run unit tests only if all quick tests pass
    if passed == total:
        if run_unit_tests():
            passed += 1
        total += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print(f"⏱️  Test Duration: {duration:.1f} seconds")
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ QUICK TESTS PASSED - Ready for deployment!")
        return 0
    else:
        print("❌ QUICK TESTS FAILED - Fix issues before deployment!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 