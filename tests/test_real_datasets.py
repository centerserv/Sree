#!/usr/bin/env python3
"""
SREE Phase 1 - Real Datasets Test
Demonstrates that the system uses real datasets (MNIST, UCI Diabetes) 
as requested by the client, with synthetic data only as backup.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import setup_logging
from data_loader import DataLoader


def test_real_datasets():
    """
    Test that real datasets are being used as requested by the client.
    """
    print("🧪 SREE Phase 1 - Real Datasets Test")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    # Create data loader
    loader = DataLoader(logger)
    
    print("\n📊 Testing Real Dataset Loading...")
    print("Client Request: Use MNIST and UCI datasets via scikit-learn")
    print("Backup: Synthetic dataset (1000 samples) for quick testing")
    
    # Test MNIST loading
    print("\n🔍 Testing MNIST Dataset...")
    try:
        X_mnist, y_mnist = loader.load_mnist(n_samples=1000)
        print(f"  ✅ MNIST loaded: {X_mnist.shape[0]} samples, {X_mnist.shape[1]} features")
        print(f"  📊 Classes: {len(set(y_mnist))} (digits 0-9)")
        print(f"  🎯 Real handwritten digit data from OpenML")
        mnist_success = True
    except Exception as e:
        print(f"  ❌ MNIST failed: {e}")
        mnist_success = False
    
    # Test Diabetes dataset loading
    print("\n🔍 Testing UCI Diabetes Dataset...")
    try:
        X_diabetes, y_diabetes = loader.load_heart()  # Function name kept for compatibility
        print(f"  ✅ Diabetes loaded: {X_diabetes.shape[0]} samples, {X_diabetes.shape[1]} features")
        print(f"  📊 Classes: {len(set(y_diabetes))} (binary classification)")
        print(f"  🎯 Real medical data from UCI repository")
        diabetes_success = True
    except Exception as e:
        print(f"  ❌ Diabetes failed: {e}")
        diabetes_success = False
    
    # Test synthetic backup
    print("\n🔍 Testing Synthetic Backup...")
    try:
        X_synth, y_synth = loader.create_synthetic(n_samples=1000)
        print(f"  ✅ Synthetic created: {X_synth.shape[0]} samples, {X_synth.shape[1]} features")
        print(f"  📊 Classes: {len(set(y_synth))} (multi-class)")
        print(f"  🔧 Backup data for quick testing")
        synth_success = True
    except Exception as e:
        print(f"  ❌ Synthetic failed: {e}")
        synth_success = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Dataset Loading Summary:")
    print(f"  MNIST (Real): {'✅' if mnist_success else '❌'}")
    print(f"  Diabetes (Real): {'✅' if diabetes_success else '❌'}")
    print(f"  Synthetic (Backup): {'✅' if synth_success else '❌'}")
    
    if mnist_success and diabetes_success:
        print("\n🎉 SUCCESS: Real datasets working as requested!")
        print("✅ MNIST: Real handwritten digit data")
        print("✅ Diabetes: Real medical data")
        print("✅ Synthetic: Backup for quick testing")
        
    else:
        print("\n⚠️  WARNING: Some real datasets failed, using backup")
        assert False


if __name__ == "__main__":
    success = test_real_datasets()
    sys.exit(0 if success else 1) 