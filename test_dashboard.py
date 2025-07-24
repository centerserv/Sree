#!/usr/bin/env python3
"""
SREE Dashboard Test Script
Quick test to verify all components are working before running the dashboard.
"""

import sys
import traceback

def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_sree_components():
    """Test SREE component imports."""
    print("\n🔍 Testing SREE components...")
    
    try:
        from data_loader import DataLoader
        print("✅ DataLoader")
    except ImportError as e:
        print(f"❌ DataLoader import failed: {e}")
        return False
    
    try:
        from layers.pattern import PatternValidator
        print("✅ PatternValidator")
    except ImportError as e:
        print(f"❌ PatternValidator import failed: {e}")
        return False
    
    try:
        from layers.presence import PresenceValidator
        print("✅ PresenceValidator")
    except ImportError as e:
        print(f"❌ PresenceValidator import failed: {e}")
        return False
    
    try:
        from layers.permanence import PermanenceValidator
        print("✅ PermanenceValidator")
    except ImportError as e:
        print(f"❌ PermanenceValidator import failed: {e}")
        return False
    
    try:
        from layers.logic import LogicValidator
        print("✅ LogicValidator")
    except ImportError as e:
        print(f"❌ LogicValidator import failed: {e}")
        return False
    
    try:
        from loop.trust_loop import TrustUpdateLoop
        print("✅ TrustUpdateLoop")
    except ImportError as e:
        print(f"❌ TrustUpdateLoop import failed: {e}")
        return False
    
    return True

def test_dashboard():
    """Test dashboard import."""
    print("\n🔍 Testing dashboard...")
    
    try:
        import dashboard
        print("✅ Dashboard imports successfully")
    except ImportError as e:
        print(f"❌ Dashboard import failed: {e}")
        return False
    
    return True

def test_visualizations():
    """Test visualization files."""
    print("\n🔍 Testing visualization files...")
    
    import os
    from pathlib import Path
    
    plots_dir = Path("plots")
    required_files = [
        "fig1.png",
        "fig2.png", 
        "ablation_visualization.png",
        "performance_comparison.png",
        "fig4.pdf",
        "fig4_preview.png"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = plots_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        print("   Run 'python3 visualization.py' to generate missing visualizations")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality."""
    print("\n🔍 Testing data loading...")
    
    try:
        from data_loader import DataLoader
        dl = DataLoader()
        
        # Test synthetic data
        X, y = dl.create_synthetic(100)
        print(f"✅ Synthetic data: {X.shape}, {y.shape}")
        
        # Test MNIST loading
        X, y = dl.load_mnist()
        print(f"✅ MNIST data: {X.shape}, {y.shape}")
        
        # Test Heart data
        X, y = dl.load_heart()
        print(f"✅ Heart data: {X.shape}, {y.shape}")
        
        # Test CIFAR-10 loading (small sample)
        X, y = dl.load_cifar10(100)
        print(f"✅ CIFAR-10 data: {X.shape}, {y.shape}")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_sree_analysis():
    """Test basic SREE analysis."""
    print("\n🔍 Testing SREE analysis...")
    
    try:
        from data_loader import DataLoader
        from layers.pattern import PatternValidator
        from sklearn.model_selection import cross_val_score
        
        # Load small dataset
        dl = DataLoader()
        X, y = dl.create_synthetic(200)
        
        # Test pattern layer
        pv = PatternValidator()
        cv_scores = cross_val_score(pv.model, X, y, cv=3, scoring='accuracy')
        print(f"✅ Pattern layer CV: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Test training
        pv.train(X, y)
        predictions = pv.predict(X)
        print(f"✅ Pattern layer training: {len(predictions)} predictions")
        
    except Exception as e:
        print(f"❌ SREE analysis failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 SREE Dashboard Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("SREE Components", test_sree_components),
        ("Dashboard", test_dashboard),
        ("Visualizations", test_visualizations),
        ("Data Loading", test_data_loading),
        ("SREE Analysis", test_sree_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work correctly.")
        print("\n🚀 To start the dashboard:")
        print("   streamlit run dashboard.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Common fixes:")
        print("   pip install -r requirements.txt")
        print("   python3 visualization.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 