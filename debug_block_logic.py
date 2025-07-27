#!/usr/bin/env python3
"""
Debug Block Logic
Understand why block count is still consistent despite removing hardcoded conditions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from data_loader import DataLoader
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from loop.trust_loop import TrustUpdateLoop
from config import setup_logging

def debug_block_logic():
    """Debug the block creation logic to understand consistency."""
    logger = setup_logging()
    logger.info("Debugging block creation logic...")
    
    # Set deterministic random seeds
    np.random.seed(42)
    import random
    random.seed(42)
    
    # Load heart disease dataset
    try:
        df = pd.read_csv('heart_disease_dataset_new.csv')
        X = df.drop('target', axis=1).values
        y = df['target'].values
        logger.info(f"Loaded heart disease dataset: {X.shape[0]} samples, {X.shape[1]} features")
    except FileNotFoundError:
        # Fallback to synthetic data
        data_loader = DataLoader()
        X, y = data_loader.load_synthetic_credit_risk()
        logger.info(f"Using synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize SREE components
    pattern_validator = PatternValidator()
    presence_validator = PresenceValidator()
    permanence_validator = PermanenceValidator()
    logic_validator = LogicValidator()
    
    # Create trust loop with validators
    trust_loop = TrustUpdateLoop(validators=[
        pattern_validator,
        presence_validator,
        permanence_validator,
        logic_validator
    ])
    
    # Train pattern validator
    logger.info("Training Pattern validator...")
    train_results = pattern_validator.train(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Run PPP loop
    logger.info("Running PPP loop...")
    final_results = trust_loop.run_ppp_loop(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Get individual layer results
    pattern_trust = pattern_validator.validate(X_test_scaled, y_test)
    presence_trust = presence_validator.validate(X_test_scaled, y_test)
    permanence_trust = permanence_validator.validate(X_test_scaled, y_test)
    logic_trust = logic_validator.validate(X_test_scaled, y_test)
    
    # Analyze trust score patterns
    print("=" * 80)
    print("TRUST SCORE ANALYSIS")
    print("=" * 80)
    
    print(f"Pattern Trust - Mean: {np.mean(pattern_trust):.4f}, Std: {np.std(pattern_trust):.4f}")
    print(f"Presence Trust - Mean: {np.mean(presence_trust):.4f}, Std: {np.std(presence_trust):.4f}")
    print(f"Permanence Trust - Mean: {np.mean(permanence_trust):.4f}, Std: {np.std(permanence_trust):.4f}")
    print(f"Logic Trust - Mean: {np.mean(logic_trust):.4f}, Std: {np.std(logic_trust):.4f}")
    
    # Check trust score distribution
    print(f"\nTrust Score Percentiles:")
    print(f"Pattern: 25th={np.percentile(pattern_trust, 25):.4f}, 75th={np.percentile(pattern_trust, 75):.4f}")
    print(f"Presence: 25th={np.percentile(presence_trust, 25):.4f}, 75th={np.percentile(presence_trust, 75):.4f}")
    print(f"Permanence: 25th={np.percentile(permanence_trust, 25):.4f}, 75th={np.percentile(permanence_trust, 75):.4f}")
    print(f"Logic: 25th={np.percentile(logic_trust, 25):.4f}, 75th={np.percentile(logic_trust, 75):.4f}")
    
    # Get block count and analyze permanence layer
    permanence_stats = permanence_validator.get_ledger_statistics()
    block_count = permanence_stats.get('total_blocks', 0)
    
    print(f"\nBlock Count: {block_count}")
    print(f"Current Block Size: {len(permanence_validator._current_block)}")
    print(f"Block Size Threshold: {permanence_validator._block_size}")
    print(f"Block Size // 2: {permanence_validator._block_size // 2}")
    
    # Analyze the permanence validation logic
    print(f"\nPermanence Validation Logic Analysis:")
    
    # Simulate the validation logic
    validation_records = permanence_validator._create_validation_records(X_test_scaled, y_test)
    trust_scores = permanence_validator._calculate_consistency_scores(X_test_scaled, validation_records)
    
    print(f"Trust Scores - Mean: {np.mean(trust_scores):.4f}, Std: {np.std(trust_scores):.4f}")
    print(f"Trust Scores - Min: {np.min(trust_scores):.4f}, Max: {np.max(trust_scores):.4f}")
    
    # Check the conditions that trigger block creation
    high_confidence_mask = trust_scores > np.percentile(trust_scores, 75)
    low_confidence_mask = trust_scores < np.percentile(trust_scores, 25)
    
    print(f"\nConfidence Masks:")
    print(f"High Confidence Samples: {np.sum(high_confidence_mask)}")
    print(f"Low Confidence Samples: {np.sum(low_confidence_mask)}")
    print(f"High Confidence Threshold: {np.sum(high_confidence_mask) >= permanence_validator._block_size // 2}")
    print(f"Low Confidence Threshold: {np.sum(low_confidence_mask) >= permanence_validator._block_size // 3}")
    
    # Check the new dynamic conditions
    trust_std = np.std(trust_scores)
    trust_mean = np.mean(trust_scores)
    
    print(f"\nDynamic Block Creation Conditions:")
    print(f"Trust Std > 0.1: {trust_std > 0.1} (std = {trust_std:.4f})")
    print(f"Trust Mean > 0.8: {trust_mean > 0.8} (mean = {trust_mean:.4f})")
    print(f"Would Create Block: {trust_std > 0.1 or trust_mean > 0.8}")
    
    # Check if there are enough records for block creation
    print(f"\nBlock Creation Thresholds:")
    print(f"Current Block Size: {len(permanence_validator._current_block)}")
    print(f"Required for Dynamic: {permanence_validator._block_size // 2}")
    print(f"Required for Minimum: {permanence_validator._block_size}")
    print(f"Has Enough for Dynamic: {len(permanence_validator._current_block) >= permanence_validator._block_size // 2}")
    print(f"Has Enough for Minimum: {len(permanence_validator._current_block) >= permanence_validator._block_size}")
    
    # Analyze the ledger
    print(f"\nLedger Analysis:")
    print(f"Total Blocks: {len(permanence_validator._ledger)}")
    for i, block in enumerate(permanence_validator._ledger):
        print(f"  Block {i+1}: {len(block)} records, Hash: {block[0]['block_hash'][:20]}...")
    
    return block_count

if __name__ == "__main__":
    debug_block_logic() 