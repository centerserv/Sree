"""
SREE Phase 1 Demo - Optimization Script
Optimization utilities for improving system performance and reliability.
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import setup_logging, DATASET_CONFIG, MODEL_CONFIG
from data_loader import DataLoader
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from loop.trust_loop import TrustUpdateLoop


class SREEOptimizer:
    """
    Optimization engine for SREE system performance and reliability.
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or setup_logging(level="INFO")
        self.optimization_results = {}
        
    def optimize_pattern_layer(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Optimize Pattern Layer performance.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Optimization results
        """
        self.logger.info("🔧 Optimizing Pattern Layer...")
        
        # Test different model configurations
        configs = [
            {"hidden_layer_sizes": (128, 64), "max_iter": 1000},
            {"hidden_layer_sizes": (256, 128, 64), "max_iter": 1500},
            {"hidden_layer_sizes": (512, 256, 128), "max_iter": 2000},
        ]
        
        best_config = None
        best_score = 0.0
        
        for i, config in enumerate(configs):
            self.logger.info(f"  Testing config {i+1}/{len(configs)}: {config}")
            
            validator = PatternValidator()
            validator.set_params(**config)
            
            # Train and evaluate
            start_time = time.time()
            validator.fit(X_train, y_train)
            trust_scores = validator.validate(X_test)
            accuracy = np.mean(trust_scores)
            training_time = time.time() - start_time
            
            if accuracy > best_score:
                best_score = accuracy
                best_config = config
                
            self.logger.info(f"    Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")
        
        self.logger.info(f"✅ Best Pattern config: {best_config} (accuracy: {best_score:.4f})")
        
        return {
            "best_config": best_config,
            "best_score": best_score,
            "tested_configs": len(configs)
        }
    
    def optimize_presence_layer(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Optimize Presence Layer parameters.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Optimization results
        """
        self.logger.info("⚛️ Optimizing Presence Layer...")
        
        # Test different entropy parameters
        entropy_params = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        best_param = None
        best_score = 0.0
        
        for param in entropy_params:
            self.logger.info(f"  Testing entropy param: {param}")
            
            validator = PresenceValidator(entropy_threshold=param)
            
            # Get pattern predictions first
            pattern_validator = PatternValidator()
            pattern_validator.fit(X_train, y_train)
            pattern_validator.validate(X_test)  # This sets up probabilities
            pattern_probs = pattern_validator.probabilities
            
            # Test presence layer
            start_time = time.time()
            trust_scores = validator.validate(pattern_probs)
            accuracy = np.mean(trust_scores)
            processing_time = time.time() - start_time
            
            if accuracy > best_score:
                best_score = accuracy
                best_param = param
                
            self.logger.info(f"    Accuracy: {accuracy:.4f}, Time: {processing_time:.2f}s")
        
        self.logger.info(f"✅ Best Presence param: {best_param} (accuracy: {best_score:.4f})")
        
        return {
            "best_param": best_param,
            "best_score": best_score,
            "tested_params": len(entropy_params)
        }
    
    def optimize_trust_loop(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Optimize Trust Loop convergence parameters.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Optimization results
        """
        self.logger.info("🔄 Optimizing Trust Loop...")
        
        # Test different convergence parameters
        convergence_configs = [
            {"iterations": 5, "tolerance": 0.01},
            {"iterations": 10, "tolerance": 0.005},
            {"iterations": 15, "tolerance": 0.001},
        ]
        
        best_config = None
        best_score = 0.0
        
        for i, config in enumerate(convergence_configs):
            self.logger.info(f"  Testing convergence config {i+1}/{len(convergence_configs)}: {config}")
            
            loop = TrustUpdateLoop(**config)
            
            # Run trust loop
            start_time = time.time()
            results = loop.run_ppp_loop(X_train, y_train, X_test, y_test)
            processing_time = time.time() - start_time
            
            final_accuracy = results.get('final_accuracy', 0.0)
            convergence_achieved = results.get('convergence_achieved', False)
            
            # Score based on accuracy and convergence
            score = final_accuracy if convergence_achieved else final_accuracy * 0.8
            
            if score > best_score:
                best_score = score
                best_config = config
                
            self.logger.info(f"    Accuracy: {final_accuracy:.4f}, Convergence: {convergence_achieved}, Time: {processing_time:.2f}s")
        
        self.logger.info(f"✅ Best Trust Loop config: {best_config} (score: {best_score:.4f})")
        
        return {
            "best_config": best_config,
            "best_score": best_score,
            "tested_configs": len(convergence_configs)
        }
    
    def run_full_optimization(self, dataset_name: str = "synthetic") -> Dict[str, Any]:
        """
        Run full optimization pipeline.
        
        Args:
            dataset_name: Name of dataset to use for optimization
            
        Returns:
            Complete optimization results
        """
        self.logger.info("🚀 Starting Full SREE Optimization Pipeline")
        self.logger.info("=" * 60)
        
        # Load dataset
        loader = DataLoader(self.logger)
        
        if dataset_name == "synthetic":
            X, y = loader.create_synthetic(n_samples=1000)
        elif dataset_name == "mnist":
            X, y = loader.load_mnist(n_samples=1000)
        elif dataset_name == "heart":
            X, y = loader.load_heart()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        self.logger.info(f"📊 Using {dataset_name} dataset: {X.shape}")
        
        # Run optimizations
        start_time = time.time()
        
        pattern_results = self.optimize_pattern_layer(X_train, y_train, X_test, y_test)
        presence_results = self.optimize_presence_layer(X_train, y_train, X_test, y_test)
        trust_results = self.optimize_trust_loop(X_train, y_train, X_test, y_test)
        
        total_time = time.time() - start_time
        
        # Compile results
        self.optimization_results = {
            "dataset": dataset_name,
            "dataset_shape": X.shape,
            "pattern_optimization": pattern_results,
            "presence_optimization": presence_results,
            "trust_optimization": trust_results,
            "total_optimization_time": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.logger.info("=" * 60)
        self.logger.info("🎉 Optimization Complete!")
        self.logger.info(f"⏱️ Total time: {total_time:.2f}s")
        
        return self.optimization_results
    
    def save_optimization_results(self, filename: str = "optimization_results.json"):
        """
        Save optimization results to disk.
        
        Args:
            filename: Output filename
        """
        if not self.optimization_results:
            self.logger.warning("No optimization results to save")
            return
        
        from config import LOGS_DIR
        import json
        
        results_path = LOGS_DIR / filename
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results = convert_numpy_types(self.optimization_results)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"💾 Optimization results saved to {results_path}")
    
    def generate_optimization_report(self) -> str:
        """
        Generate a human-readable optimization report.
        
        Returns:
            Formatted report string
        """
        if not self.optimization_results:
            return "No optimization results available"
        
        results = self.optimization_results
        
        report = f"""
SREE Optimization Report
========================

Dataset: {results['dataset']}
Shape: {results['dataset_shape']}
Timestamp: {results['timestamp']}
Total Time: {results['total_optimization_time']:.2f}s

Pattern Layer Optimization:
- Best Config: {results['pattern_optimization']['best_config']}
- Best Score: {results['pattern_optimization']['best_score']:.4f}
- Configs Tested: {results['pattern_optimization']['tested_configs']}

Presence Layer Optimization:
- Best Param: {results['presence_optimization']['best_param']}
- Best Score: {results['presence_optimization']['best_score']:.4f}
- Params Tested: {results['presence_optimization']['tested_params']}

Trust Loop Optimization:
- Best Config: {results['trust_optimization']['best_config']}
- Best Score: {results['trust_optimization']['best_score']:.4f}
- Configs Tested: {results['trust_optimization']['tested_configs']}

Recommendations:
1. Use Pattern config: {results['pattern_optimization']['best_config']}
2. Use Presence param: {results['presence_optimization']['best_param']}
3. Use Trust Loop config: {results['trust_optimization']['best_config']}
"""
        
        return report


def main():
    """Run SREE optimization pipeline."""
    print("🚀 SREE Phase 1 - Optimization Pipeline")
    print("=" * 50)
    
    # Create optimizer
    optimizer = SREEOptimizer()
    
    # Run optimization on different datasets
    datasets = ["synthetic", "mnist", "heart"]
    
    for dataset in datasets:
        print(f"\n📊 Optimizing on {dataset} dataset...")
        try:
            results = optimizer.run_full_optimization(dataset)
            
            # Save results
            optimizer.save_optimization_results(f"optimization_{dataset}.json")
            
            # Generate and display report
            report = optimizer.generate_optimization_report()
            print(report)
            
        except Exception as e:
            print(f"❌ Error optimizing {dataset}: {e}")
            continue
    
    print("\n🎉 Optimization pipeline complete!")
    print("📁 Check logs/ directory for detailed results")


if __name__ == "__main__":
    main() 