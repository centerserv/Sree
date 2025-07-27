#!/usr/bin/env python3
"""
Test Multi-Dataset Block Count
Verify that block count varies across different datasets, showing true dynamic behavior.
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

def test_multi_dataset_block_count():
    """Test that block count varies across different datasets."""
    logger = setup_logging()
    logger.info("Testing block count across different datasets...")
    
    # Set deterministic random seeds
    np.random.seed(42)
    import random
    random.seed(42)
    
    datasets = []
    
    # Dataset 1: Heart Disease
    try:
        df = pd.read_csv('heart_disease_dataset_new.csv')
        X = df.drop('target', axis=1).values
        y = df['target'].values
        datasets.append(('Heart Disease', X, y))
        logger.info(f"Added Heart Disease dataset: {X.shape[0]} samples, {X.shape[1]} features")
    except FileNotFoundError:
        logger.warning("Heart disease dataset not found")
    
    # Dataset 2: Synthetic Credit Risk
    try:
        data_loader = DataLoader()
        X, y = data_loader.load_synthetic_credit_risk()
        datasets.append(('Synthetic Credit Risk', X, y))
        logger.info(f"Added Synthetic Credit Risk dataset: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        logger.warning(f"Could not load synthetic dataset: {e}")
    
    # Dataset 3: Small synthetic dataset
    try:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        datasets.append(('Small Synthetic', X, y))
        logger.info(f"Added Small Synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        logger.warning(f"Could not create small synthetic dataset: {e}")
    
    # Dataset 4: Large synthetic dataset
    try:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
        datasets.append(('Large Synthetic', X, y))
        logger.info(f"Added Large Synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        logger.warning(f"Could not create large synthetic dataset: {e}")
    
    if not datasets:
        logger.error("No datasets available for testing")
        return False
    
    results = []
    
    for dataset_name, X, y in datasets:
        logger.info(f"Testing dataset: {dataset_name}")
        
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
        train_results = pattern_validator.train(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Run PPP loop
        final_results = trust_loop.run_ppp_loop(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Get metrics
        accuracy = final_results.get('final_accuracy', 0.0)
        trust = final_results.get('final_trust', 0.0)
        
        # Get entropy from presence layer
        presence_stats = presence_validator.get_entropy_statistics()
        entropy = presence_stats.get('mean_entropy', 0.0)
        
        # Get block count from permanence layer
        permanence_stats = permanence_validator.get_ledger_statistics()
        block_count = permanence_stats.get('total_blocks', 0)
        
        results.append({
            'dataset': dataset_name,
            'samples': X.shape[0],
            'features': X.shape[1],
            'accuracy': accuracy,
            'trust': trust,
            'entropy': entropy,
            'block_count': block_count
        })
        
        logger.info(f"Dataset {dataset_name}: {block_count} blocks")
    
    # Display results
    print("=" * 80)
    print("MULTI-DATASET BLOCK COUNT TEST RESULTS")
    print("=" * 80)
    
    for result in results:
        print(f"Dataset: {result['dataset']}")
        print(f"  Samples: {result['samples']}, Features: {result['features']}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Trust Score: {result['trust']:.4f}")
        print(f"  Entropy: {result['entropy']:.4f}")
        print(f"  Block Count: {result['block_count']}")
        print()
    
    # Check if block counts vary
    block_counts = [r['block_count'] for r in results]
    unique_block_counts = set(block_counts)
    
    print("=" * 80)
    print("DYNAMIC BEHAVIOR ANALYSIS")
    print("=" * 80)
    print(f"Block Counts: {block_counts}")
    print(f"Unique Block Counts: {unique_block_counts}")
    print(f"Number of Datasets: {len(datasets)}")
    print(f"Number of Unique Block Counts: {len(unique_block_counts)}")
    print(f"Is Dynamic Across Datasets: {'✅ YES' if len(unique_block_counts) > 1 else '❌ NO'}")
    
    if len(unique_block_counts) > 1:
        print("🎉 SUCCESS: Block count varies across different datasets!")
        print("This demonstrates true dynamic behavior based on data characteristics.")
        return True
    else:
        print("⚠️  WARNING: Block count is the same across all datasets.")
        print("This suggests the block creation logic may still be too deterministic.")
        return False

if __name__ == "__main__":
    success = test_multi_dataset_block_count()
    sys.exit(0 if success else 1) 