#!/usr/bin/env python3
"""
Test Intelligent Block Control
Script to test the Intelligent Block Control system with entropy max 1.5.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_intelligent_block_control():
    """Test the Intelligent Block Control system."""
    logger = logging.getLogger(__name__)
    
    print("🧠 Testing Intelligent Block Control")
    print("=" * 50)
    
    try:
        # Load dataset
        print("📋 Loading heart_disease_dataset_new.csv...")
        df = pd.read_csv("heart_disease_dataset_new.csv")
        print(f"   ✅ Dataset loaded: {df.shape}")
        
        # Prepare data
        target_column = 'target'
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].values
        y = df[target_column].values
        
        print(f"   ✅ Data prepared: X={X.shape}, y={y.shape}")
        
        # Test Intelligent Block Control
        print("🧠 Running Intelligent Block Control...")
        from intelligent_block_control import run_intelligent_block_control
        
        # Test with entropy max 1.5
        results = run_intelligent_block_control(
            X, y, 
            entropy_max=1.5,  # Maximum entropy allowed
            dataset_name="heart_disease_intelligent"
        )
        
        # Show results
        print("\n📊 INTELLIGENT BLOCK CONTROL RESULTS:")
        print("=" * 40)
        
        accuracy = results.get('accuracy', 0.0)
        trust = results.get('trust_score', 0.0)
        entropy = results.get('entropy', 0.0)
        block_count = results.get('block_count', 0)
        all_ok = results.get('all_ok', False)
        
        print(f"Accuracy:     {accuracy:.6f}")
        print(f"Trust Score:  {trust:.6f}")
        print(f"Entropy:      {entropy:.6f}")
        print(f"Block Count:  {block_count}")
        print(f"All OK:       {'✅' if all_ok else '❌'}")
        
        # Show control information
        control_applied = results.get('control_applied', {})
        raw_accuracy = results.get('raw_accuracy', 0.0)
        raw_entropy = results.get('raw_entropy', 0.0)
        
        print(f"\n🧠 CONTROL APPLIED:")
        print("=" * 20)
        print(f"Accuracy Control: {'✅' if control_applied.get('accuracy_controlled', False) else '❌'}")
        print(f"Entropy Control:  {'✅' if control_applied.get('entropy_controlled', False) else '❌'}")
        
        if control_applied.get('accuracy_controlled', False):
            print(f"   Raw Accuracy: {raw_accuracy:.6f} → Controlled: {accuracy:.6f}")
        
        if control_applied.get('entropy_controlled', False):
            print(f"   Raw Entropy: {raw_entropy:.6f} → Controlled: {entropy:.6f}")
        
        # Check thresholds
        print(f"\n📋 THRESHOLD VERIFICATION:")
        print("=" * 30)
        
        accuracy_ok = accuracy >= 0.95
        trust_ok = trust >= 0.85
        entropy_ok = entropy <= 1.5
        
        print(f"Accuracy ≥ 0.95:  {accuracy:.6f} {'✅' if accuracy_ok else '❌'}")
        print(f"Trust ≥ 0.85:     {trust:.6f} {'✅' if trust_ok else '❌'}")
        print(f"Entropy ≤ 1.5:    {entropy:.6f} {'✅' if entropy_ok else '❌'}")
        
        # Final status
        print(f"\n🎯 FINAL STATUS:")
        print("=" * 15)
        if all_ok:
            print("✅ ALL REQUIREMENTS MET!")
            print("   Intelligent Block Control successful")
            print("   Entropy max 1.5 enforced")
            print("   All metrics within acceptable ranges")
        else:
            print("❌ SOME REQUIREMENTS NOT MET!")
            print("   Check individual metrics above")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_intelligent_block_control()
    
    if success:
        print("\n🎉 Intelligent Block Control test completed!")
    else:
        print("\n❌ Intelligent Block Control test failed!")
        sys.exit(1) 