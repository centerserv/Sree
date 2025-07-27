#!/usr/bin/env python3
"""
Test Dashboard Block Count
Quick test to verify the dashboard analysis produces 3 blocks.
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

def test_dashboard_analysis():
    """Test the same analysis that the dashboard runs."""
    logger = setup_logging()
    logger.info("Testing dashboard analysis for block count...")
    
    # Set deterministic random seeds (same as verification script)
    np.random.seed(42)
    import random
    random.seed(42)
    
    # Load heart disease dataset (same as dashboard)
    try:
        df = pd.read_csv('heart_disease_dataset_new.csv')
        X = df.drop('target', axis=1).values
        y = df['target'].values
        logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    except FileNotFoundError:
        # Fallback to synthetic data
        data_loader = DataLoader()
        X, y = data_loader.load_synthetic_credit_risk()
        logger.info(f"Using synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data (same as dashboard)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (same as dashboard)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize SREE components (same as verification script)
    pattern_validator = PatternValidator()
    presence_validator = PresenceValidator()
    permanence_validator = PermanenceValidator()
    logic_validator = LogicValidator()
    
    # Create trust loop with validators (same as verification script)
    trust_loop = TrustUpdateLoop(validators=[
        pattern_validator,
        presence_validator,
        permanence_validator,
        logic_validator
    ])
    
    # Train pattern validator (same as dashboard)
    logger.info("Training Pattern validator...")
    train_results = pattern_validator.train(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Run PPP loop (same as dashboard)
    logger.info("Running PPP loop...")
    ppp_results = trust_loop.run_ppp_loop(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Get individual layer results (same as dashboard)
    pattern_trust = pattern_validator.validate(X_test_scaled, y_test)
    presence_trust = presence_validator.validate(X_test_scaled, y_test)
    permanence_trust = permanence_validator.validate(X_test_scaled, y_test)
    logic_trust = logic_validator.validate(X_test_scaled, y_test)
    
    # Calculate metrics (same as dashboard)
    accuracy = ppp_results.get('final_accuracy', 0.0)
    trust = ppp_results.get('final_trust', 0.0)
    
    # Get entropy from presence layer (same as dashboard)
    presence_stats = presence_validator.get_entropy_statistics()
    entropy = presence_stats.get('mean_entropy', 0.0)
    
    # Get block count from permanence layer (same as dashboard)
    permanence_stats = permanence_validator.get_ledger_statistics()
    block_count = permanence_stats.get('total_blocks', 0)
    
    # Display results
    print("=" * 60)
    print("DASHBOARD ANALYSIS TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Trust Score: {trust:.4f}")
    print(f"Entropy: {entropy:.4f}")
    print(f"Block Count: {block_count}")
    print(f"Expected Blocks: 3")
    print(f"Block Count Match: {'✅ YES' if block_count == 3 else '❌ NO'}")
    print("=" * 60)
    
    if block_count == 3:
        print("✅ SUCCESS: Dashboard analysis produces 3 blocks!")
        return True
    else:
        print("❌ ISSUE: Dashboard analysis produces wrong block count!")
        return False

if __name__ == "__main__":
    success = test_dashboard_analysis()
    sys.exit(0 if success else 1) 