#!/usr/bin/env python3
"""
Test Dynamic Block Count
Verify that the block count is now dynamic and not hardcoded to 3.
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

def test_dynamic_block_count():
    """Test that block count is dynamic and not hardcoded."""
    logger = setup_logging()
    logger.info("Testing dynamic block count...")
    
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
    
    # Calculate metrics
    accuracy = final_results.get('final_accuracy', 0.0)
    trust = final_results.get('final_trust', 0.0)
    
    # Get entropy from presence layer
    presence_stats = presence_validator.get_entropy_statistics()
    entropy = presence_stats.get('mean_entropy', 0.0)
    
    # Get block count from permanence layer
    permanence_stats = permanence_validator.get_ledger_statistics()
    block_count = permanence_stats.get('total_blocks', 0)
    
    # Display results
    print("=" * 60)
    print("DYNAMIC BLOCK COUNT TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Trust Score: {trust:.4f}")
    print(f"Entropy: {entropy:.4f}")
    print(f"Block Count: {block_count}")
    print(f"Block Count Type: {type(block_count)}")
    print(f"Is Block Count Dynamic: {'✅ YES' if block_count != 3 else '❌ NO'}")
    print("=" * 60)
    
    # Check if block count varies with different random seeds
    print("\nTesting block count variation with different seeds...")
    
    block_counts = []
    for seed in [42, 123, 456, 789, 999]:
        # Reset validators
        pattern_validator.reset()
        presence_validator.reset()
        permanence_validator.reset()
        logic_validator.reset()
        
        # Set new seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Run analysis
        train_results = pattern_validator.train(X_train_scaled, y_train, X_test_scaled, y_test)
        final_results = trust_loop.run_ppp_loop(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Get block count
        permanence_stats = permanence_validator.get_ledger_statistics()
        block_count = permanence_stats.get('total_blocks', 0)
        block_counts.append(block_count)
        
        print(f"Seed {seed}: {block_count} blocks")
    
    print(f"\nBlock Counts: {block_counts}")
    print(f"Unique Block Counts: {set(block_counts)}")
    print(f"Is Dynamic: {'✅ YES' if len(set(block_counts)) > 1 else '❌ NO'}")
    
    if len(set(block_counts)) > 1:
        print("🎉 SUCCESS: Block count is now dynamic and varies with different conditions!")
        return True
    else:
        print("⚠️  WARNING: Block count is still not varying. May need further investigation.")
        return False

if __name__ == "__main__":
    success = test_dynamic_block_count()
    sys.exit(0 if success else 1) 