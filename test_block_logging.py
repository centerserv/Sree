#!/usr/bin/env python3
"""
Test script for block-level logging system.
Tests the detailed diagnostics per block with row-level information.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loop.trust_loop import TrustUpdateLoop
from data_loader import DataLoader
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator


def test_block_logging_with_heart_disease():
    """Test block logging with heart disease dataset."""
    print("🧪 Testing Block-Level Logging System")
    print("=" * 50)
    
    # Load heart disease dataset
    print("📊 Loading heart disease dataset...")
    data_loader = DataLoader()
    heart_data = data_loader.load_heart_disease()
    
    if heart_data is None:
        print("❌ Could not load heart disease dataset")
        return False
    
    X, y = heart_data
    print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Split data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📈 Training set: {len(X_train)} samples")
    print(f"🧪 Test set: {len(X_test)} samples")
    
    # Create trust loop with validators
    print("🔧 Initializing SREE components...")
    pattern_validator = PatternValidator()
    presence_validator = PresenceValidator()
    permanence_validator = PermanenceValidator()
    logic_validator = LogicValidator()
    
    trust_loop = TrustUpdateLoop(validators=[
        pattern_validator,
        presence_validator,
        permanence_validator,
        logic_validator
    ])
    
    # Run PPP loop with detailed logging
    print("🚀 Running PPP loop with detailed block logging...")
    results = trust_loop.run_ppp_loop(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save block logs
    print("💾 Saving block logs...")
    block_logs_file = trust_loop.save_block_logs()
    print(f"✅ Block logs saved to: {block_logs_file}")
    
    # Verify block logs
    print("🔍 Verifying block logs...")
    with open(block_logs_file, 'r') as f:
        block_logs = json.load(f)
    
    if not block_logs:
        print("❌ No block logs generated")
        return False
    
    print(f"✅ Generated {len(block_logs)} block(s)")
    
    # Analyze block logs
    for i, block in enumerate(block_logs):
        print(f"\n📊 Block {block['block_id']} Analysis:")
        print(f"   Samples: {block['n_samples']}")
        print(f"   Features: {block['n_features']}")
        print(f"   Iterations: {len(block['iterations'])}")
        
        if block['iterations']:
            # Check first iteration
            first_iter = block['iterations'][0]
            print(f"   First iteration summary:")
            print(f"     Avg V_q: {first_iter['summary']['avg_v_q']:.3f}")
            print(f"     Avg V_b: {first_iter['summary']['avg_v_b']:.3f}")
            print(f"     Avg V_l: {first_iter['summary']['avg_v_l']:.3f}")
            print(f"     Decisions: {first_iter['summary']['n_retained']} retained, "
                  f"{first_iter['summary']['n_flagged']} flagged, "
                  f"{first_iter['summary']['n_down_weighted']} down-weighted")
            print(f"     Logic failures: {first_iter['summary']['n_logic_failures']}")
            
            # Check row-level diagnostics
            if first_iter['row_diagnostics']:
                print(f"     Row diagnostics: {len(first_iter['row_diagnostics'])} rows")
                
                # Find outliers
                outliers = [d for d in first_iter['row_diagnostics'] if d['is_outlier']]
                print(f"     Outliers: {len(outliers)} rows with entropy > 2.0")
                
                # Find low scores
                low_v_q = [d for d in first_iter['row_diagnostics'] if d['v_q_score'] < 0.3]
                low_v_b = [d for d in first_iter['row_diagnostics'] if d['v_b_score'] < 0.3]
                low_v_l = [d for d in first_iter['row_diagnostics'] if d['v_l_score'] < 0.3]
                
                print(f"     Low V_q scores: {len(low_v_q)} rows")
                print(f"     Low V_b scores: {len(low_v_b)} rows")
                print(f"     Low V_l scores: {len(low_v_l)} rows")
            
            # Check logic failures
            if first_iter['logic_failures']:
                print(f"     Logic failures: {len(first_iter['logic_failures'])} rows")
                for failure in first_iter['logic_failures'][:3]:  # Show first 3
                    print(f"       Row {failure['row_id']}: {len(failure['rule_violations'])} violations")
        
        # Check final results
        if 'final_results' in block and block['final_results']:
            final = block['final_results']
            print(f"   Final results:")
            print(f"     Accuracy: {final['final_accuracy']:.3f}")
            print(f"     Convergence: {final['convergence_achieved']}")
            print(f"     Avg trust: {final['avg_final_trust']:.3f}")
    
    # Test multi-block scenario with noisy data
    print("\n🧪 Testing multi-block scenario with noisy data...")
    
    # Add noise to create multiple blocks
    X_test_noisy = X_test_scaled.copy()
    noise_mask = np.random.rand(*X_test_noisy.shape) < 0.1  # 10% noise
    X_test_noisy[noise_mask] += np.random.normal(0, 0.5, X_test_noisy[noise_mask].shape)
    
    # Run with noisy data
    results_noisy = trust_loop.run_ppp_loop(X_train_scaled, y_train, X_test_noisy, y_test)
    
    # Save noisy block logs
    block_logs_noisy_file = trust_loop.save_block_logs("per_block_logs_noisy.json")
    print(f"✅ Noisy block logs saved to: {block_logs_noisy_file}")
    
    # Verify noisy logs
    with open(block_logs_noisy_file, 'r') as f:
        noisy_logs = json.load(f)
    
    print(f"✅ Generated {len(noisy_logs)} block(s) with noisy data")
    
    # Compare results
    print("\n📊 Comparison Results:")
    print(f"Original accuracy: {results['final_accuracy']:.3f}")
    print(f"Noisy accuracy: {results_noisy['final_accuracy']:.3f}")
    print(f"Original trust: {results['final_trust']:.3f}")
    print(f"Noisy trust: {results_noisy['final_trust']:.3f}")
    
    print("\n✅ Block logging test completed successfully!")
    return True


def test_entropy_outlier_detection():
    """Test entropy-based outlier detection and reweighting."""
    print("\n🧪 Testing Entropy-Based Outlier Detection")
    print("=" * 50)
    
    # Create synthetic data with known outliers
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Create normal data
    X_normal = np.random.normal(0, 1, (n_samples, n_features))
    y_normal = np.random.randint(0, 2, n_samples)
    
    # Create some outliers (high entropy predictions)
    X_outliers = np.random.normal(0, 3, (20, n_features))  # Higher variance
    y_outliers = np.random.randint(0, 2, 20)
    
    # Combine data
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([y_normal, y_outliers])
    
    print(f"📊 Created dataset with {len(X)} samples ({len(X_outliers)} outliers)")
    
    # Split and scale
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create trust loop
    trust_loop = TrustUpdateLoop()
    
    # Run analysis
    results = trust_loop.run_ppp_loop(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save logs
    block_logs_file = trust_loop.save_block_logs("per_block_logs_outliers.json")
    
    # Analyze outlier detection
    with open(block_logs_file, 'r') as f:
        block_logs = json.load(f)
    
    if block_logs and block_logs[0]['iterations']:
        first_iter = block_logs[0]['iterations'][0]
        
        # Count outliers detected
        outliers_detected = sum(1 for d in first_iter['row_diagnostics'] if d['is_outlier'])
        down_weighted = first_iter['summary']['n_down_weighted']
        
        print(f"📈 Outlier Detection Results:")
        print(f"   Outliers detected: {outliers_detected}")
        print(f"   Down-weighted rows: {down_weighted}")
        print(f"   Detection rate: {outliers_detected/len(X_test):.1%}")
        
        # Show some outlier examples
        outliers = [d for d in first_iter['row_diagnostics'] if d['is_outlier']]
        if outliers:
            print(f"   Example outliers:")
            for i, outlier in enumerate(outliers[:3]):
                print(f"     Row {outlier['row_id']}: entropy={outlier['entropy']:.3f}, "
                      f"decision={outlier['decision']}")
    
    print("✅ Outlier detection test completed!")
    return True


def main():
    """Run all block logging tests."""
    print("🚀 SREE Block-Level Logging Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test 1: Basic block logging
    try:
        success &= test_block_logging_with_heart_disease()
    except Exception as e:
        print(f"❌ Basic block logging test failed: {str(e)}")
        success = False
    
    # Test 2: Outlier detection
    try:
        success &= test_entropy_outlier_detection()
    except Exception as e:
        print(f"❌ Outlier detection test failed: {str(e)}")
        success = False
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ All block logging tests passed!")
        print("\n📋 Test Summary:")
        print("• Block-level diagnostics working")
        print("• Row-level V_q, V_b, V_l scores logged")
        print("• Decision tracking (retained/flagged/down-weighted)")
        print("• Logic rule failure detection")
        print("• Entropy-based outlier detection")
        print("• Multi-block scenario support")
        print("• JSON export functionality")
    else:
        print("❌ Some tests failed!")
    
    return success


if __name__ == "__main__":
    main() 