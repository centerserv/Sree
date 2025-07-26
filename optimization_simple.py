"""
SREE Phase 1 Demo - Simple Optimization Script
Simplified optimization for testing.
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import setup_logging
from data_loader import DataLoader
from layers.pattern import PatternValidator
from loop.trust_loop import TrustUpdateLoop


def simple_optimization():
    """Run simple optimization test."""
    print("🚀 SREE Simple Optimization Test")
    print("=" * 40)
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    # Load dataset
    loader = DataLoader(logger)
    X, y = loader.create_synthetic(n_samples=500)
    X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
    
    print(f"📊 Dataset shape: {X.shape}")
    
    # Test Pattern Layer
    print("\n🔧 Testing Pattern Layer...")
    pattern_validator = PatternValidator()
    
    start_time = time.time()
    pattern_validator.fit(X_train, y_train)
    trust_scores = pattern_validator.validate(X_test)
    accuracy = np.mean(trust_scores)
    training_time = time.time() - start_time
    
    print(f"  ✅ Pattern accuracy: {accuracy:.4f}")
    print(f"  ⏱️ Training time: {training_time:.2f}s")
    
    # Test Trust Loop
    print("\n🔄 Testing Trust Loop...")
    loop = TrustUpdateLoop(iterations=3)
    
    start_time = time.time()
    results = loop.run_ppp_loop(X_train, y_train, X_test, y_test)
    processing_time = time.time() - start_time
    
    final_accuracy = results.get('final_accuracy', 0.0)
    convergence_achieved = results.get('convergence_achieved', False)
    
    print(f"  ✅ Final accuracy: {final_accuracy:.4f}")
    print(f"  🔄 Convergence: {convergence_achieved}")
    print(f"  ⏱️ Processing time: {processing_time:.2f}s")
    
    print("\n🎉 Simple optimization test complete!")
    
    return {
        "pattern_accuracy": accuracy,
        "final_accuracy": final_accuracy,
        "convergence": convergence_achieved,
        "pattern_time": training_time,
        "total_time": processing_time
    }


if __name__ == "__main__":
    results = simple_optimization()
    print(f"\n📊 Results: {results}") 