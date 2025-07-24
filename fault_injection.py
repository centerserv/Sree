#!/usr/bin/env python3
"""
SREE Phase 1 - Fault Injection Testing Framework
Tests system resilience under 10-15% label corruption.
Target: T ≈ 0.85-0.88 trust score under corruption.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

from config import setup_logging, TEST_CONFIG
from data_loader import DataLoader
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from loop.trust_loop import TrustUpdateLoop


class FaultInjector:
    """
    Fault injection framework for testing SREE resilience.
    
    Implements various corruption strategies:
    - Label flipping (random, targeted)
    - Feature corruption
    - Distribution shifts
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize fault injector.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or setup_logging()
        self.corruption_results = {}
        
    def inject_label_corruption(self, y: np.ndarray, corruption_rate: float = 0.15, 
                               strategy: str = "random", random_state: int = 42) -> Tuple[np.ndarray, Dict]:
        """
        Inject label corruption into the dataset.
        
        Args:
            y: Original labels
            corruption_rate: Fraction of labels to corrupt (0.0-1.0)
            strategy: Corruption strategy ("random", "targeted", "adversarial")
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (corrupted_labels, corruption_info)
        """
        np.random.seed(random_state)
        n_samples = len(y)
        n_corrupt = int(n_samples * corruption_rate)
        
        self.logger.info(f"Injecting {corruption_rate:.1%} label corruption using {strategy} strategy")
        
        # Create copy of original labels
        y_corrupted = y.copy()
        corruption_info = {
            "strategy": strategy,
            "corruption_rate": corruption_rate,
            "n_corrupted": n_corrupt,
            "n_total": n_samples,
            "corrupted_indices": []
        }
        
        if strategy == "random":
            # Random label flipping
            corrupt_indices = np.random.choice(n_samples, n_corrupt, replace=False)
            for idx in corrupt_indices:
                original_label = y_corrupted[idx]
                # Flip to a different random class
                possible_labels = [l for l in np.unique(y) if l != original_label]
                y_corrupted[idx] = np.random.choice(possible_labels)
                corruption_info["corrupted_indices"].append(idx)
                
        elif strategy == "targeted":
            # Target specific classes for corruption
            unique_labels = np.unique(y)
            target_classes = unique_labels[:len(unique_labels)//2]  # Target first half of classes
            
            # Find samples from target classes
            target_indices = np.where(np.isin(y, target_classes))[0]
            corrupt_indices = np.random.choice(target_indices, 
                                             min(n_corrupt, len(target_indices)), 
                                             replace=False)
            
            for idx in corrupt_indices:
                original_label = y_corrupted[idx]
                # Flip to non-target class
                non_target_labels = [l for l in unique_labels if l not in target_classes]
                y_corrupted[idx] = np.random.choice(non_target_labels)
                corruption_info["corrupted_indices"].append(idx)
                
        elif strategy == "adversarial":
            # Adversarial corruption: flip to most confusing class
            unique_labels = np.unique(y)
            label_counts = np.bincount(y, minlength=len(unique_labels))
            
            for _ in range(n_corrupt):
                # Find least frequent class to create confusion
                least_frequent = np.argmin(label_counts)
                # Find sample from most frequent class
                most_frequent = np.argmax(label_counts)
                most_frequent_indices = np.where(y == most_frequent)[0]
                
                if len(most_frequent_indices) > 0:
                    idx = np.random.choice(most_frequent_indices)
                    y_corrupted[idx] = least_frequent
                    corruption_info["corrupted_indices"].append(idx)
        
        # Update corruption info
        corruption_info["n_actual_corrupted"] = len(corruption_info["corrupted_indices"])
        corruption_info["actual_corruption_rate"] = corruption_info["n_actual_corrupted"] / n_samples
        
        self.logger.info(f"Corruption complete: {corruption_info['n_actual_corrupted']} labels corrupted "
                        f"({corruption_info['actual_corruption_rate']:.1%})")
        
        return y_corrupted, corruption_info
    
    def inject_feature_corruption(self, X: np.ndarray, corruption_rate: float = 0.10,
                                 strategy: str = "noise", random_state: int = 42) -> Tuple[np.ndarray, Dict]:
        """
        Inject feature corruption into the dataset.
        
        Args:
            X: Original features
            corruption_rate: Fraction of features to corrupt
            strategy: Corruption strategy ("noise", "zero", "outlier")
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (corrupted_features, corruption_info)
        """
        np.random.seed(random_state)
        n_samples, n_features = X.shape
        n_corrupt_features = int(n_features * corruption_rate)
        
        self.logger.info(f"Injecting {corruption_rate:.1%} feature corruption using {strategy} strategy")
        
        X_corrupted = X.copy()
        corruption_info = {
            "strategy": strategy,
            "corruption_rate": corruption_rate,
            "n_corrupted_features": n_corrupt_features,
            "corrupted_feature_indices": []
        }
        
        # Select features to corrupt
        corrupt_feature_indices = np.random.choice(n_features, n_corrupt_features, replace=False)
        corruption_info["corrupted_feature_indices"] = corrupt_feature_indices.tolist()
        
        if strategy == "noise":
            # Add Gaussian noise
            noise_std = np.std(X) * 0.5
            for feat_idx in corrupt_feature_indices:
                noise = np.random.normal(0, noise_std, n_samples)
                X_corrupted[:, feat_idx] += noise
                
        elif strategy == "zero":
            # Zero out features
            for feat_idx in corrupt_feature_indices:
                X_corrupted[:, feat_idx] = 0
                
        elif strategy == "outlier":
            # Replace with extreme values
            for feat_idx in corrupt_feature_indices:
                feature_mean = np.mean(X[:, feat_idx])
                feature_std = np.std(X[:, feat_idx])
                outliers = np.random.normal(feature_mean + 3*feature_std, feature_std, n_samples)
                X_corrupted[:, feat_idx] = outliers
        
        self.logger.info(f"Feature corruption complete: {n_corrupt_features} features corrupted")
        return X_corrupted, corruption_info


class FaultInjectionTester:
    """
    Comprehensive fault injection testing framework for SREE.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize fault injection tester.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or setup_logging()
        self.fault_injector = FaultInjector(self.logger)
        self.results = {}
        
    def test_clean_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Test performance on clean (uncorrupted) data.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Dictionary with clean performance metrics
        """
        self.logger.info("Testing clean performance baseline...")
        
        # Create validators
        pattern_validator = PatternValidator()
        presence_validator = PresenceValidator()
        permanence_validator = PermanenceValidator()
        logic_validator = LogicValidator()
        
        # Create trust loop
        trust_loop = TrustUpdateLoop(
            validators=[pattern_validator, presence_validator, permanence_validator, logic_validator],
            iterations=10
        )
        
        # Test on clean data
        # Split data for training and testing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = trust_loop.run_ppp_loop(X_train, y_train, X_test, y_test)
        
        clean_metrics = {
            "accuracy": results["final_accuracy"],
            "trust": results["final_trust"],
            "convergence": results["convergence_achieved"],
            "iterations": len(results["iterations"]),
            "pattern_trust": results["iterations"][-1]["pattern_trust"] if results["iterations"] else 0.0,
            "presence_trust": results["iterations"][-1]["presence_trust"] if results["iterations"] else 0.0,
            "permanence_trust": results["iterations"][-1]["permanence_trust"] if results["iterations"] else 0.0,
            "logic_trust": results["iterations"][-1]["logic_trust"] if results["iterations"] else 0.0
        }
        
        self.logger.info(f"Clean performance: accuracy={clean_metrics['accuracy']:.3f}, "
                        f"trust={clean_metrics['trust']:.3f}")
        
        return clean_metrics
    
    def test_label_corruption_resilience(self, X: np.ndarray, y: np.ndarray, 
                                       corruption_rates: List[float] = None) -> Dict[str, Any]:
        """
        Test resilience under various label corruption rates.
        
        Args:
            X: Feature matrix
            y: Label vector
            corruption_rates: List of corruption rates to test
            
        Returns:
            Dictionary with corruption resilience results
        """
        if corruption_rates is None:
            corruption_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
        
        self.logger.info(f"Testing label corruption resilience with rates: {corruption_rates}")
        
        corruption_results = {}
        
        for rate in corruption_rates:
            self.logger.info(f"\n--- Testing {rate:.1%} label corruption ---")
            
            # Inject corruption
            y_corrupted, corruption_info = self.fault_injector.inject_label_corruption(
                y, corruption_rate=rate, strategy="random"
            )
            
            # Test performance on corrupted data
            corrupted_metrics = self.test_clean_performance(X, y_corrupted)
            
            # Store results
            corruption_results[f"{rate:.1%}"] = {
                "corruption_info": corruption_info,
                "performance": corrupted_metrics,
                "degradation": {
                    "accuracy_loss": self.results.get("clean", {}).get("accuracy", 0) - corrupted_metrics["accuracy"],
                    "trust_loss": self.results.get("clean", {}).get("trust", 0) - corrupted_metrics["trust"]
                }
            }
            
            self.logger.info(f"Corruption {rate:.1%}: accuracy={corrupted_metrics['accuracy']:.3f}, "
                           f"trust={corrupted_metrics['trust']:.3f}")
        
        return corruption_results
    
    def test_feature_corruption_resilience(self, X: np.ndarray, y: np.ndarray,
                                         corruption_rates: List[float] = None) -> Dict[str, Any]:
        """
        Test resilience under feature corruption.
        
        Args:
            X: Feature matrix
            y: Label vector
            corruption_rates: List of corruption rates to test
            
        Returns:
            Dictionary with feature corruption results
        """
        if corruption_rates is None:
            corruption_rates = [0.05, 0.10, 0.15]
        
        self.logger.info(f"Testing feature corruption resilience with rates: {corruption_rates}")
        
        feature_results = {}
        
        for rate in corruption_rates:
            self.logger.info(f"\n--- Testing {rate:.1%} feature corruption ---")
            
            # Inject feature corruption
            X_corrupted, corruption_info = self.fault_injector.inject_feature_corruption(
                X, corruption_rate=rate, strategy="noise"
            )
            
            # Test performance on corrupted data
            corrupted_metrics = self.test_clean_performance(X_corrupted, y)
            
            # Store results
            feature_results[f"{rate:.1%}"] = {
                "corruption_info": corruption_info,
                "performance": corrupted_metrics,
                "degradation": {
                    "accuracy_loss": self.results.get("clean", {}).get("accuracy", 0) - corrupted_metrics["accuracy"],
                    "trust_loss": self.results.get("clean", {}).get("trust", 0) - corrupted_metrics["trust"]
                }
            }
            
            self.logger.info(f"Feature corruption {rate:.1%}: accuracy={corrupted_metrics['accuracy']:.3f}, "
                           f"trust={corrupted_metrics['trust']:.3f}")
        
        return feature_results
    
    def run_comprehensive_fault_test(self, dataset_name: str = "mnist") -> Dict[str, Any]:
        """
        Run comprehensive fault injection testing.
        
        Args:
            dataset_name: Name of dataset to test ("mnist", "diabetes", "synthetic")
            
        Returns:
            Dictionary with comprehensive test results
        """
        self.logger.info(f"Starting comprehensive fault injection testing on {dataset_name} dataset")
        
        # Load dataset
        data_loader = DataLoader(self.logger)
        
        if dataset_name == "mnist":
            X, y = data_loader.load_mnist(n_samples=2000)
        elif dataset_name == "heart":
            X, y = data_loader.load_heart()
        else:
            X, y = data_loader.create_synthetic(n_samples=2000)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = data_loader.preprocess_data(X, y)
        
        # Use test set for fault injection testing
        X_test, y_test = X_test, y_test
        
        # Test clean performance first
        self.logger.info("=== CLEAN PERFORMANCE BASELINE ===")
        clean_metrics = self.test_clean_performance(X_test, y_test)
        self.results["clean"] = clean_metrics
        
        # Test label corruption resilience
        self.logger.info("\n=== LABEL CORRUPTION RESILIENCE ===")
        label_results = self.test_label_corruption_resilience(X_test, y_test)
        self.results["label_corruption"] = label_results
        
        # Test feature corruption resilience
        self.logger.info("\n=== FEATURE CORRUPTION RESILIENCE ===")
        feature_results = self.test_feature_corruption_resilience(X_test, y_test)
        self.results["feature_corruption"] = feature_results
        
        # Generate summary
        summary = self.generate_fault_test_summary()
        
        # Save results
        self.save_fault_test_results(dataset_name)
        
        return summary
    
    def generate_fault_test_summary(self) -> Dict[str, Any]:
        """
        Generate summary of fault injection test results.
        
        Returns:
            Dictionary with test summary
        """
        clean_metrics = self.results.get("clean", {})
        label_results = self.results.get("label_corruption", {})
        feature_results = self.results.get("feature_corruption", {})
        
        # Calculate resilience metrics
        resilience_metrics = {
            "clean_accuracy": clean_metrics.get("accuracy", 0),
            "clean_trust": clean_metrics.get("trust", 0),
            "target_trust_under_15pct": None,
            "resilience_score": 0
        }
        
        # Find trust score under 15% corruption
        if "15.0%" in label_results:
            resilience_metrics["target_trust_under_15pct"] = label_results["15.0%"]["performance"]["trust"]
        
        # Calculate resilience score (how well system maintains performance under corruption)
        if resilience_metrics["target_trust_under_15pct"]:
            resilience_score = resilience_metrics["target_trust_under_15pct"] / resilience_metrics["clean_trust"]
            resilience_metrics["resilience_score"] = resilience_score
        
        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "resilience_metrics": resilience_metrics,
            "label_corruption_summary": {
                rate: {
                    "accuracy": results.get("performance", {}).get("accuracy", 0.0),
                    "trust": results.get("performance", {}).get("trust", 0.0),
                    "accuracy_loss": results.get("degradation", {}).get("accuracy_loss", 0.0),
                    "trust_loss": results.get("degradation", {}).get("trust_loss", 0.0)
                }
                for rate, results in label_results.items()
            },
            "feature_corruption_summary": {
                rate: {
                    "accuracy": results.get("performance", {}).get("accuracy", 0.0),
                    "trust": results.get("performance", {}).get("trust", 0.0),
                    "accuracy_loss": results.get("degradation", {}).get("accuracy_loss", 0.0),
                    "trust_loss": results.get("degradation", {}).get("trust_loss", 0.0)
                }
                for rate, results in feature_results.items()
            }
        }
        
        return summary
    
    def save_fault_test_results(self, dataset_name: str):
        """
        Save fault injection test results to file.
        
        Args:
            dataset_name: Name of dataset tested
        """
        results_path = Path("logs") / f"fault_injection_results_{dataset_name}.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Fault injection results saved to {results_path}")


def main():
    """
    Main function to run fault injection testing.
    """
    print("🧪 SREE Phase 1 - Fault Injection Testing")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    # Create fault injection tester
    tester = FaultInjectionTester(logger)
    
    # Test on different datasets
    datasets = ["mnist", "heart", "synthetic"]
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n🔍 Testing {dataset.upper()} dataset...")
        try:
            summary = tester.run_comprehensive_fault_test(dataset)
            all_results[dataset] = summary
            
            # Print summary
            resilience = summary["resilience_metrics"]
            print(f"  ✅ Clean accuracy: {resilience['clean_accuracy']:.3f}")
            print(f"  ✅ Clean trust: {resilience['clean_trust']:.3f}")
            print(f"  ✅ Trust under 15% corruption: {resilience['target_trust_under_15pct']:.3f}")
            print(f"  ✅ Resilience score: {resilience['resilience_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to test {dataset}: {e}")
            print(f"  ❌ {dataset} test failed: {e}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 FAULT INJECTION TESTING SUMMARY")
    print("=" * 50)
    
    for dataset, results in all_results.items():
        resilience = results["resilience_metrics"]
        print(f"\n{dataset.upper()}:")
        print(f"  Clean Trust: {resilience['clean_trust']:.3f}")
        print(f"  Trust @ 15% corruption: {resilience['target_trust_under_15pct']:.3f}")
        print(f"  Resilience: {resilience['resilience_score']:.3f}")
        
        # Check if target achieved
        if resilience['target_trust_under_15pct'] and resilience['target_trust_under_15pct'] >= 0.85:
            print(f"  🎯 TARGET ACHIEVED: Trust ≥ 0.85 under 15% corruption")
        else:
            print(f"  ⚠️  TARGET MISSED: Trust < 0.85 under 15% corruption")
    
    return all_results


if __name__ == "__main__":
    main() 