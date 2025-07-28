#!/usr/bin/env python3
"""
Test Dashboard Local
Script to test the dashboard locally before deployment.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_dashboard_local():
    """Test the dashboard locally to verify accuracy display."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing Dashboard Locally")
    logger.info("=" * 50)
    
    try:
        # Test 1: Import dashboard
        logger.info("📋 Test 1: Importing dashboard...")
        from dashboard import SREEDashboard
        logger.info("✅ Dashboard imported successfully")
        
        # Test 2: Create dashboard instance
        logger.info("📋 Test 2: Creating dashboard instance...")
        dashboard = SREEDashboard()
        logger.info("✅ Dashboard instance created successfully")
        
        # Test 3: Load heart disease dataset
        logger.info("📋 Test 3: Loading heart disease dataset...")
        df = pd.read_csv("heart_disease_dataset_new.csv")
        logger.info(f"✅ Dataset loaded: {df.shape}")
        
        # Test 4: Prepare data
        logger.info("📋 Test 4: Preparing data...")
        target_column = 'target'
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].values
        y = df[target_column].values
        
        logger.info(f"✅ Data prepared: X={X.shape}, y={y.shape}")
        logger.info(f"   Features: {len(feature_columns)}")
        logger.info(f"   Classes: {len(np.unique(y))}")
        
        # Test 5: Run SREE analysis
        logger.info("📋 Test 5: Running SREE analysis...")
        results = dashboard.run_sree_analysis(X, y)
        
        logger.info("📊 Analysis Results:")
        logger.info(f"   Accuracy: {results.get('accuracy', 0.0):.6f}")
        logger.info(f"   Trust Score: {results.get('trust_score', 0.0):.6f}")
        logger.info(f"   Entropy: {results.get('entropy', 0.0):.6f}")
        logger.info(f"   Block Count: {results.get('block_count', 0)}")
        logger.info(f"   Accuracy OK: {results.get('accuracy_ok', False)}")
        logger.info(f"   Trust OK: {results.get('trust_ok', False)}")
        logger.info(f"   Entropy OK: {results.get('entropy_ok', False)}")
        
        # Test 6: Verify thresholds
        logger.info("📋 Test 6: Verifying thresholds...")
        accuracy = results.get('accuracy', 0.0)
        trust = results.get('trust_score', 0.0)
        entropy = results.get('entropy', 0.0)
        
        accuracy_ok = accuracy >= 0.95
        trust_ok = trust >= 0.85
        entropy_ok = entropy <= 1.5
        
        logger.info(f"   Accuracy {accuracy:.6f} ≥ 0.95: {'✅' if accuracy_ok else '❌'}")
        logger.info(f"   Trust {trust:.6f} ≥ 0.85: {'✅' if trust_ok else '❌'}")
        logger.info(f"   Entropy {entropy:.6f} ≤ 1.5: {'✅' if entropy_ok else '❌'}")
        
        # Test 7: Test display function (mock)
        logger.info("📋 Test 7: Testing display function...")
        
        # Mock display test
        display_test_results = {
            'accuracy': accuracy,
            'trust_score': trust,
            'entropy': entropy,
            'block_count': results.get('block_count', 0),
            'accuracy_ok': accuracy_ok,
            'trust_ok': trust_ok,
            'entropy_ok': entropy_ok
        }
        
        logger.info("📊 Display Test Results:")
        logger.info(f"   Accuracy: {display_test_results['accuracy']:.3f}")
        logger.info(f"   Trust Score: {display_test_results['trust_score']:.3f}")
        logger.info(f"   Entropy: {display_test_results['entropy']:.3f}")
        logger.info(f"   Block Count: {display_test_results['block_count']}")
        
        # Test 8: Verify all tests passed
        logger.info("📋 Test 8: Final verification...")
        
        all_tests_passed = (
            accuracy_ok and 
            trust_ok and 
            entropy_ok and 
            results.get('block_count', 0) > 0
        )
        
        if all_tests_passed:
            logger.info("🎉 ALL TESTS PASSED! Dashboard is ready for deployment.")
            logger.info("✅ Accuracy meets client requirements (≥ 0.95)")
            logger.info("✅ Trust Score meets client requirements (≥ 0.85)")
            logger.info("✅ Entropy meets client requirements (≤ 1.5)")
            logger.info("✅ Block count is valid (> 0)")
            return True
        else:
            logger.error("❌ SOME TESTS FAILED! Dashboard needs fixes before deployment.")
            if not accuracy_ok:
                logger.error(f"   ❌ Accuracy {accuracy:.6f} < 0.95")
            if not trust_ok:
                logger.error(f"   ❌ Trust {trust:.6f} < 0.85")
            if not entropy_ok:
                logger.error(f"   ❌ Entropy {entropy:.6f} > 1.5")
            if results.get('block_count', 0) <= 0:
                logger.error(f"   ❌ Block count {results.get('block_count', 0)} <= 0")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during local dashboard test: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_dashboard_local()
    
    if success:
        print("\n🎉 DASHBOARD LOCAL TEST PASSED!")
        print("✅ Ready for deployment")
        sys.exit(0)
    else:
        print("\n❌ DASHBOARD LOCAL TEST FAILED!")
        print("❌ Fix issues before deployment")
        sys.exit(1) 