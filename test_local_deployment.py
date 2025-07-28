#!/usr/bin/env python3
"""
Local Deployment Test Script
Tests the centralized block creation system locally before deployment.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """Setup logging for local testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    logger = setup_logging()
    logger.info("🧪 Testing imports...")
    
    try:
        # Test SREE components
        from data_loader import DataLoader
        from layers.pattern import PatternValidator
        from layers.presence import PresenceValidator
        from layers.permanence import PermanenceValidator
        from layers.logic import LogicValidator
        from loop.trust_loop import TrustUpdateLoop
        
        # Test centralized system
        from block_creation_system import run_block_creation_system, run_single_analysis
        
        logger.info("✅ All imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test if datasets can be loaded."""
    logger = setup_logging()
    logger.info("🧪 Testing data loading...")
    
    try:
        from data_loader import DataLoader
        
        loader = DataLoader()
        
        # Test heart disease dataset
        X, y = loader.load_heart()
        logger.info(f"✅ Heart dataset loaded: {X.shape}, {y.shape}")
        
        # Test synthetic credit risk dataset
        X_syn, y_syn = loader.load_synthetic_credit_risk()
        logger.info(f"✅ Synthetic dataset loaded: {X_syn.shape}, {y_syn.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading error: {e}")
        return False

def test_centralized_system():
    """Test the centralized block creation system."""
    logger = setup_logging()
    logger.info("🧪 Testing centralized block creation system...")
    
    try:
        from data_loader import DataLoader
        from block_creation_system import run_single_analysis
        
        # Load test dataset
        loader = DataLoader()
        X, y = loader.load_heart()
        
        # Test single analysis
        logger.info("Running single analysis test...")
        results = run_single_analysis(X, y, dataset_name="test_heart")
        
        # Verify results structure
        required_keys = ['accuracy', 'trust_score', 'entropy', 'block_count', 
                        'accuracy_ok', 'trust_ok', 'entropy_ok']
        
        for key in required_keys:
            if key not in results:
                logger.error(f"❌ Missing key in results: {key}")
                return False
        
        # Check thresholds
        accuracy = results['accuracy']
        trust_score = results['trust_score']
        entropy = results['entropy']
        
        logger.info(f"📊 Test Results:")
        logger.info(f"   Accuracy: {accuracy:.6f} (≥0.95) {'✅' if results['accuracy_ok'] else '❌'}")
        logger.info(f"   Trust Score: {trust_score:.6f} (≥0.85) {'✅' if results['trust_ok'] else '❌'}")
        logger.info(f"   Entropy: {entropy:.6f} (≤1.5) {'✅' if results['entropy_ok'] else '❌'}")
        logger.info(f"   Block Count: {results['block_count']}")
        
        logger.info("✅ Centralized system test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Centralized system error: {e}")
        return False

def test_dashboard_integration():
    """Test if dashboard can use the centralized system."""
    logger = setup_logging()
    logger.info("🧪 Testing dashboard integration...")
    
    try:
        # Import dashboard components
        from dashboard import SREEDashboard
        
        # Create dashboard instance
        dashboard = SREEDashboard()
        
        # Load test data
        from data_loader import DataLoader
        loader = DataLoader()
        X, y = loader.load_heart()
        
        # Test dashboard analysis
        logger.info("Running dashboard analysis test...")
        results = dashboard.run_sree_analysis(X, y)
        
        # Verify results
        if 'error' in results:
            logger.error(f"❌ Dashboard analysis error: {results['error']}")
            return False
        
        logger.info(f"📊 Dashboard Results:")
        logger.info(f"   Accuracy: {results.get('accuracy', 0):.6f}")
        logger.info(f"   Trust Score: {results.get('trust_score', 0):.6f}")
        logger.info(f"   Entropy: {results.get('entropy', 0):.6f}")
        logger.info(f"   Block Count: {results.get('block_count', 0)}")
        
        logger.info("✅ Dashboard integration test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dashboard integration error: {e}")
        return False

def test_main_integration():
    """Test if main.py can use the centralized system."""
    logger = setup_logging()
    logger.info("🧪 Testing main.py integration...")
    
    try:
        # Import main components
        from block_creation_system import run_block_creation_system
        from data_loader import DataLoader
        
        # Load test data
        loader = DataLoader()
        X, y = loader.load_heart()
        
        # Prepare dataset data
        dataset_data = {
            'X': X,
            'y': y,
            'name': 'test_heart'
        }
        
        # Test main system (single test for speed)
        logger.info("Running main system test...")
        results = run_block_creation_system('test_heart', dataset_data, n_tests=1)
        
        # Verify results structure
        if 'acceptable_ranges' not in results:
            logger.error("❌ Missing acceptable_ranges in results")
            return False
        
        logger.info(f"📊 Main System Results:")
        logger.info(f"   Acceptable Ranges: {results['acceptable_ranges']}")
        logger.info(f"   Individual Tests: {len(results.get('individual_tests', []))}")
        
        logger.info("✅ Main integration test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Main integration error: {e}")
        return False

def test_thresholds():
    """Test if thresholds are correctly applied."""
    logger = setup_logging()
    logger.info("🧪 Testing threshold validation...")
    
    try:
        from block_creation_system import run_single_analysis
        from data_loader import DataLoader
        
        # Load test data
        loader = DataLoader()
        X, y = loader.load_heart()
        
        # Run analysis
        results = run_single_analysis(X, y, dataset_name="threshold_test")
        
        # Check thresholds
        accuracy_ok = results['accuracy_ok']
        trust_ok = results['trust_ok']
        entropy_ok = results['entropy_ok']
        
        # Verify threshold values
        expected_accuracy_threshold = 0.95
        expected_trust_threshold = 0.85
        expected_entropy_threshold = 1.5
        
        accuracy_check = results['accuracy'] >= expected_accuracy_threshold
        trust_check = results['trust_score'] >= expected_trust_threshold
        entropy_check = results['entropy'] <= expected_entropy_threshold
        
        logger.info(f"📊 Threshold Validation:")
        logger.info(f"   Accuracy: {results['accuracy']:.6f} ≥ {expected_accuracy_threshold} {'✅' if accuracy_check else '❌'}")
        logger.info(f"   Trust: {results['trust_score']:.6f} ≥ {expected_trust_threshold} {'✅' if trust_check else '❌'}")
        logger.info(f"   Entropy: {results['entropy']:.6f} ≤ {expected_entropy_threshold} {'✅' if entropy_check else '❌'}")
        
        # Verify consistency
        if accuracy_ok != accuracy_check:
            logger.error("❌ Accuracy threshold validation inconsistent")
            return False
        
        if trust_ok != trust_check:
            logger.error("❌ Trust threshold validation inconsistent")
            return False
        
        if entropy_ok != entropy_check:
            logger.error("❌ Entropy threshold validation inconsistent")
            return False
        
        logger.info("✅ Threshold validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Threshold validation error: {e}")
        return False

def run_all_tests():
    """Run all local tests."""
    logger = setup_logging()
    logger.info("🚀 Starting Local Deployment Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Centralized System Test", test_centralized_system),
        ("Dashboard Integration Test", test_dashboard_integration),
        ("Main Integration Test", test_main_integration),
        ("Threshold Validation Test", test_thresholds)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Ready for deployment.")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED! Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 