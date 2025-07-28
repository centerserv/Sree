#!/usr/bin/env python3
"""
Debug Dashboard Accuracy
Script to debug why dashboard shows different accuracy than expected.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """Setup logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def debug_accuracy_issue():
    """Debug the accuracy discrepancy issue."""
    logger = setup_logging()
    
    logger.info("🔍 Debugging Dashboard Accuracy Issue")
    logger.info("=" * 50)
    
    # Load heart disease dataset
    from data_loader import DataLoader
    loader = DataLoader()
    X, y = loader.load_heart()
    
    logger.info(f"📊 Dataset loaded: X.shape={X.shape}, y.shape={y.shape}")
    
    # Test centralized system
    logger.info("\n🧪 Testing Centralized System...")
    from block_creation_system import run_single_analysis
    
    results = run_single_analysis(X, y, "debug_test")
    
    logger.info("📊 Centralized System Results:")
    logger.info(f"   Accuracy: {results.get('accuracy', 0.0):.6f}")
    logger.info(f"   Trust Score: {results.get('trust_score', 0.0):.6f}")
    logger.info(f"   Entropy: {results.get('entropy', 0.0):.6f}")
    logger.info(f"   Block Count: {results.get('block_count', 0)}")
    logger.info(f"   Accuracy OK: {results.get('accuracy_ok', False)}")
    
    # Test dashboard integration
    logger.info("\n🧪 Testing Dashboard Integration...")
    from dashboard import SREEDashboard
    
    dashboard = SREEDashboard()
    dashboard_results = dashboard.run_sree_analysis(X, y)
    
    logger.info("📊 Dashboard Results:")
    logger.info(f"   Accuracy: {dashboard_results.get('accuracy', 0.0):.6f}")
    logger.info(f"   Trust Score: {dashboard_results.get('trust_score', 0.0):.6f}")
    logger.info(f"   Entropy: {dashboard_results.get('entropy', 0.0):.6f}")
    logger.info(f"   Block Count: {dashboard_results.get('block_count', 0)}")
    logger.info(f"   Accuracy OK: {dashboard_results.get('accuracy_ok', False)}")
    
    # Compare results
    logger.info("\n🔍 Comparison:")
    accuracy_diff = abs(results.get('accuracy', 0.0) - dashboard_results.get('accuracy', 0.0))
    logger.info(f"   Accuracy Difference: {accuracy_diff:.6f}")
    
    if accuracy_diff > 0.001:
        logger.error("❌ ACCURACY MISMATCH DETECTED!")
        logger.error(f"   Centralized: {results.get('accuracy', 0.0):.6f}")
        logger.error(f"   Dashboard: {dashboard_results.get('accuracy', 0.0):.6f}")
    else:
        logger.info("✅ Accuracy values match!")
    
    # Check if results are the same object
    logger.info(f"\n🔍 Object Comparison:")
    logger.info(f"   Same object: {results is dashboard_results}")
    logger.info(f"   Results keys: {list(results.keys())}")
    logger.info(f"   Dashboard keys: {list(dashboard_results.keys())}")
    
    return results, dashboard_results

if __name__ == "__main__":
    debug_accuracy_issue() 