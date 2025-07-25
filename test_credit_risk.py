#!/usr/bin/env python3
"""
Test script for Credit Risk dataset to verify Phase 1 targets.
"""

import logging
import numpy as np
from pathlib import Path
from data_loader import DataLoader
from main import run_final_phase1_tests

def test_credit_risk_dataset():
    """
    Test the credit risk dataset to verify Phase 1 targets.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Testing Credit Risk Dataset for Phase 1 Targets")
    logger.info("=" * 60)
    
    # Create data loader
    loader = DataLoader()
    
    # Create credit risk dataset
    logger.info("Creating credit risk dataset...")
    X, y = loader.create_credit_risk_dataset(n_samples=800)
    
    # Get dataset info
    info = loader.get_dataset_info(X, y)
    logger.info(f"Dataset Info: {info}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
    
    # Create dataset dict
    dataset_data = {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "info": info
    }
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    logger.info(f"Good credit rate: {np.mean(y):.2%}")
    
    # Run tests
    logger.info("Running Phase 1 tests...")
    results = run_final_phase1_tests("credit_risk", dataset_data, n_tests=20)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("CREDIT RISK DATASET - PHASE 1 RESULTS")
    logger.info("=" * 60)
    logger.info(f"Dataset: {results['dataset']}")
    logger.info(f"Tests Run: {results['n_tests']}")
    logger.info(f"Accuracy: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
    logger.info(f"Trust Score: {results['trust']['mean']:.4f} ± {results['trust']['std']:.4f}")
    logger.info(f"Block Count: {results['block_count']['mean']:.1f} ± {results['block_count']['std']:.1f}")
    logger.info(f"Entropy: {results['entropy']['mean']:.4f} ± {results['entropy']['std']:.4f}")
    
    # Check Phase 1 requirements
    accuracy_ok = results['accuracy']['mean'] >= 0.95
    trust_ok = results['trust']['mean'] >= 0.85
    entropy_ok = 2.0 <= results['entropy']['mean'] <= 4.0
    blocks_ok = results['block_count']['mean'] > 4
    variance_ok = results['accuracy']['std'] < 0.03
    
    logger.info(f"\nPhase 1 Requirements:")
    logger.info(f"  Accuracy ≥ 95%: {'✅' if accuracy_ok else '❌'} ({results['accuracy']['mean']:.1%})")
    logger.info(f"  Trust ≥ 85%: {'✅' if trust_ok else '❌'} ({results['trust']['mean']:.1%})")
    logger.info(f"  Entropy 2-4: {'✅' if entropy_ok else '❌'} ({results['entropy']['mean']:.2f})")
    logger.info(f"  Blocks > 4: {'✅' if blocks_ok else '❌'} ({results['block_count']['mean']:.1f})")
    logger.info(f"  Variance < 3%: {'✅' if variance_ok else '❌'} ({results['accuracy']['std']:.1%})")
    
    all_met = accuracy_ok and trust_ok and entropy_ok and blocks_ok and variance_ok
    logger.info(f"  All Met: {'✅' if all_met else '❌'}")
    
    if all_met:
        logger.info("\n🎉 SUCCESS! All Phase 1 targets achieved!")
    else:
        logger.info("\n⚠️  Some Phase 1 targets not met. Review needed.")
    
    logger.info("=" * 60)
    
    return results

if __name__ == "__main__":
    test_credit_risk_dataset() 