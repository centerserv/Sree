#!/usr/bin/env python3
"""
Verify Block Count Consistency
Simple script to verify that both environments produce exactly 3 blocks
"""

import os
import json
import numpy as np
import pandas as pd
import random
import platform
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Import SREE components
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from loop.trust_loop import TrustUpdateLoop
from config import setup_logging, PPP_CONFIG

def verify_block_count_consistency():
    """Verify that the system produces exactly 3 blocks consistently"""
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting block count consistency verification...")
    
    # Set deterministic seeds
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    
    # Log environment info
    logger.info(f"Environment: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"Random seed: 42")
    
    # Load dataset
    logger.info("Loading heart disease dataset...")
    df = pd.read_csv('heart_disease_dataset_new.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples")
    
    # Initialize SREE components
    logger.info("Initializing SREE components...")
    pattern_validator = PatternValidator()
    presence_validator = PresenceValidator()
    permanence_validator = PermanenceValidator()
    logic_validator = LogicValidator()
    
    # Train pattern validator
    logger.info("Training pattern validator...")
    pattern_results = pattern_validator.train(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Run trust update loop
    logger.info("Running trust update loop...")
    trust_loop = TrustUpdateLoop(
        validators=[pattern_validator, presence_validator, permanence_validator, logic_validator]
    )
    
    # Run PPP loop
    logger.info("Running PPP loop...")
    final_results = trust_loop.run_ppp_loop(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Get permanence validator state
    permanence_state = permanence_validator.get_state()
    ledger_stats = permanence_validator.get_ledger_statistics()
    
    # Check block count
    actual_blocks = ledger_stats.get('total_blocks', 0)
    expected_blocks = 3
    
    logger.info("="*50)
    logger.info("BLOCK COUNT VERIFICATION RESULTS")
    logger.info("="*50)
    logger.info(f"Expected blocks: {expected_blocks}")
    logger.info(f"Actual blocks: {actual_blocks}")
    logger.info(f"Block count match: {actual_blocks == expected_blocks}")
    logger.info(f"Final accuracy: {final_results['final_accuracy']:.4f}")
    logger.info(f"Final trust: {final_results['final_trust']:.4f}")
    logger.info(f"Convergence: {final_results['convergence_achieved']}")
    logger.info(f"Iterations: {len(final_results['iterations'])}")
    logger.info("="*50)
    
    # Print summary
    print("\n" + "="*60)
    print("BLOCK COUNT CONSISTENCY VERIFICATION")
    print("="*60)
    print(f"Environment: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print(f"Random Seed: 42")
    print(f"Expected Blocks: {expected_blocks}")
    print(f"Actual Blocks: {actual_blocks}")
    print(f"Block Count Match: {'✅ YES' if actual_blocks == expected_blocks else '❌ NO'}")
    print(f"Final Accuracy: {final_results['final_accuracy']:.4f}")
    print(f"Final Trust: {final_results['final_trust']:.4f}")
    print(f"Convergence: {final_results['convergence_achieved']}")
    print(f"Iterations: {len(final_results['iterations'])}")
    
    if actual_blocks == expected_blocks:
        print("\n✅ SUCCESS: Both environments should now produce exactly 3 blocks!")
        print("   The code has been synchronized and the pandas Series fixes are applied.")
    else:
        print("\n❌ ISSUE: Block count mismatch detected.")
        print("   Please ensure the remote environment has the latest code version.")
    
    print("="*60)
    
    return actual_blocks == expected_blocks

if __name__ == "__main__":
    success = verify_block_count_consistency()
    exit(0 if success else 1) 