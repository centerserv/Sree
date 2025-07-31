#!/usr/bin/env python3
"""
Server Performance Test
Test the performance of SREE with server-optimized configuration
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from unified_block_creation import run_unified_block_creation

# Set server mode environment variable
os.environ['SREE_SERVER_MODE'] = 'true'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_server_performance():
    """Test performance with heart_large.csv dataset"""
    
    logger.info("🖥️ Testing Server Performance with Optimized Configuration")
    logger.info("📊 Dataset: heart_large.csv (30,000 rows)")
    logger.info("⚡ Server Mode: ENABLED")
    
    # Load dataset
    try:
        df = pd.read_csv('heart_large.csv')
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        logger.info(f"📈 Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
    except FileNotFoundError:
        logger.error("❌ heart_large.csv not found. Creating synthetic dataset...")
        # Create synthetic dataset for testing
        np.random.seed(42)
        n_samples = 30000
        n_features = 13
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        logger.info(f"📈 Synthetic dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test with server-optimized configuration
    start_time = time.time()
    
    try:
        results = run_unified_block_creation(
            X=X,
            y=y,
            accuracy_threshold=0.85,  # Relaxed for server
            trust_threshold=0.75,     # Relaxed for server
            entropy_threshold=2.2,    # Relaxed for server
            max_blocks=3,             # Minimal blocks for server
            required_consecutive_ok=1, # Stop after first good block
            dataset_name="heart_large_server_test",
            use_dashboard_config=False
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Extract final metrics
        final_accuracy = results.get('final_accuracy', 0.0)
        final_trust = results.get('final_trust', 0.0)
        final_entropy = results.get('final_entropy', 0.0)
        blocks_created = results.get('blocks_created', 0)
        
        logger.info("=" * 60)
        logger.info("🎯 SERVER PERFORMANCE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f}m)")
        logger.info(f"📊 Final Accuracy: {final_accuracy:.3f}")
        logger.info(f"🤝 Final Trust: {final_trust:.3f}")
        logger.info(f"📈 Final Entropy: {final_entropy:.3f}")
        logger.info(f"🔢 Blocks Created: {blocks_created}")
        logger.info(f"🎯 Configuration: SERVER_OPTIMIZED_CONFIG")
        logger.info("=" * 60)
        
        # Performance analysis
        if total_time < 300:  # Less than 5 minutes
            logger.info("✅ EXCELLENT: Performance under 5 minutes!")
        elif total_time < 600:  # Less than 10 minutes
            logger.info("✅ GOOD: Performance under 10 minutes")
        elif total_time < 1200:  # Less than 20 minutes
            logger.info("⚠️  ACCEPTABLE: Performance under 20 minutes")
        else:
            logger.warning("❌ SLOW: Performance over 20 minutes")
        
        # Quality analysis
        if final_accuracy >= 0.85 and final_trust >= 0.75:
            logger.info("✅ QUALITY: Metrics meet server-optimized thresholds")
        else:
            logger.warning("⚠️  QUALITY: Some metrics below server-optimized thresholds")
        
        return {
            'total_time': total_time,
            'final_accuracy': final_accuracy,
            'final_trust': final_trust,
            'final_entropy': final_entropy,
            'blocks_created': blocks_created,
            'success': True
        }
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.error(f"❌ Test failed after {total_time:.1f}s: {str(e)}")
        return {
            'total_time': total_time,
            'error': str(e),
            'success': False
        }

if __name__ == "__main__":
    test_server_performance() 