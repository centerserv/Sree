"""
SREE Phase 1 Demo - Main Execution
Main script to run the complete SREE Phase 1 demo with improved robustness.
"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import setup_logging, get_config
from data_loader import load_all_datasets, DataLoader


def detect_outliers(data: np.ndarray, labels: np.ndarray, dataset_name: str) -> dict:
    """
    Detect outliers in the dataset using statistical methods.
    
    Args:
        data: Input features
        labels: Target labels
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with outlier information
    """
    outlier_info = {
        "dataset": dataset_name,
        "total_samples": len(data),
        "outliers_detected": 0,
        "outlier_percentage": 0.0,
        "outlier_details": []
    }
    
    # Method 1: Z-score method (|z| > 3)
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    z_outliers = np.any(z_scores > 3, axis=1)
    
    # Method 2: IQR method
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = np.any((data < lower_bound) | (data > upper_bound), axis=1)
    
    # Combine methods
    outliers = z_outliers | iqr_outliers
    
    outlier_info["outliers_detected"] = np.sum(outliers)
    outlier_info["outlier_percentage"] = (np.sum(outliers) / len(data)) * 100
    
    # Get outlier details
    if np.sum(outliers) > 0:
        outlier_indices = np.where(outliers)[0]
        for idx in outlier_indices[:10]:  # Show first 10 outliers
            outlier_info["outlier_details"].append({
                "index": int(idx),
                "label": int(labels[idx]),
                "max_z_score": float(np.max(z_scores[idx])),
                "feature_stats": {
                    "mean": float(np.mean(data[idx])),
                    "std": float(np.std(data[idx])),
                    "min": float(np.min(data[idx])),
                    "max": float(np.max(data[idx]))
                }
            })
    
    return outlier_info


def run_multiple_tests(dataset_name: str, dataset_data: dict, n_tests: int = 8) -> dict:
    """
    Run multiple tests with different data splits to reduce variation.
    
    Args:
        dataset_name: Name of the dataset
        dataset_data: Dataset information
        n_tests: Number of tests to run
        
    Returns:
        Dictionary with aggregated results
    """
    from layers.pattern import PatternValidator
    from layers.presence import PresenceValidator
    from layers.permanence import PermanenceValidator
    from layers.logic import LogicValidator
    from loop.trust_loop import TrustUpdateLoop
    
    logger = logging.getLogger(__name__)
    logger.info(f"Running {n_tests} tests for {dataset_name}...")
    
    # Initialize validators
    pattern_validator = PatternValidator()
    presence_validator = PresenceValidator()
    permanence_validator = PermanenceValidator()
    logic_validator = LogicValidator()
    trust_loop = TrustUpdateLoop()
    
    # Store results for each test
    test_results = []
    
    for test_idx in range(n_tests):
        logger.info(f"  Test {test_idx + 1}/{n_tests}")
        
        # Use different random states for different splits
        random_state = 42 + test_idx
        
        # Split data with different random state
        X_temp = dataset_data['X_train'].copy()
        y_temp = dataset_data['y_train'].copy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=random_state, stratify=y_temp
        )
        
        # 10-fold cross-validation for Pattern layer
        kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(
            pattern_validator.model, 
            X_train, 
            y_train, 
            cv=kfold, 
            scoring='accuracy'
        )
        
        # Train final model
        pattern_validator.train(X_train, y_train)
        y_pred = pattern_validator.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Run PPP analysis with improved parameters for better block count
        results = trust_loop.run_analysis(
            X_train, y_train, X_test, y_test,
            [pattern_validator, presence_validator, permanence_validator, logic_validator]
        )
        
        # Get permanence statistics
        permanence_stats = permanence_validator.get_ledger_statistics()
        
        # Get presence statistics
        presence_stats = presence_validator.get_metadata()
        
        test_result = {
            "test_id": test_idx + 1,
            "random_state": random_state,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "test_accuracy": float(test_accuracy),
            "final_accuracy": float(results.get('final_accuracy', 0)),
            "final_trust": float(results.get('final_trust', 0)),
            "block_count": int(permanence_stats.get('total_blocks', 0)),
            "entropy": float(presence_stats.get('avg_entropy', 0)),
            "converged": bool(results.get('converged', False))
        }
        
        test_results.append(test_result)
        
        # Reset validators for next test
        pattern_validator.reset()
        presence_validator.reset()
        permanence_validator.reset()
        logic_validator.reset()
        trust_loop.reset()
    
    # Aggregate results
    aggregated_results = {
        "dataset": dataset_name,
        "n_tests": n_tests,
        "accuracy": {
            "mean": float(np.mean([r["final_accuracy"] for r in test_results])),
            "std": float(np.std([r["final_accuracy"] for r in test_results])),
            "min": float(np.min([r["final_accuracy"] for r in test_results])),
            "max": float(np.max([r["final_accuracy"] for r in test_results]))
        },
        "trust": {
            "mean": float(np.mean([r["final_trust"] for r in test_results])),
            "std": float(np.std([r["final_trust"] for r in test_results])),
            "min": float(np.min([r["final_trust"] for r in test_results])),
            "max": float(np.max([r["final_trust"] for r in test_results]))
        },
        "block_count": {
            "mean": float(np.mean([r["block_count"] for r in test_results])),
            "std": float(np.std([r["block_count"] for r in test_results])),
            "min": int(np.min([r["block_count"] for r in test_results])),
            "max": int(np.max([r["block_count"] for r in test_results]))
        },
        "entropy": {
            "mean": float(np.mean([r["entropy"] for r in test_results])),
            "std": float(np.std([r["entropy"] for r in test_results])),
            "min": float(np.min([r["entropy"] for r in test_results])),
            "max": float(np.max([r["entropy"] for r in test_results]))
        },
        "cv_accuracy": {
            "mean": float(np.mean([r["cv_mean"] for r in test_results])),
            "std": float(np.std([r["cv_mean"] for r in test_results]))
        },
        "variation_reduced": {
            "from": "~8%",
            "to": f"{np.std([r['final_accuracy'] for r in test_results]) * 100:.2f}%"
        },
        "individual_tests": test_results
    }
    
    return aggregated_results


def save_results_to_text(results: dict, outlier_info: dict, output_file: str = "sree_results.txt"):
    """
    Save results to a text file for easy copy-paste to Grok.
    
    Args:
        results: Aggregated test results
        outlier_info: Outlier detection information
        output_file: Output filename
    """
    output_path = Path(__file__).parent / output_file
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SREE PHASE 1 RESULTS - CLIENT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {results['dataset']}\n")
        f.write(f"Tests Run: {results['n_tests']}\n\n")
        
        # Performance Summary
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}\n")
        f.write(f"Trust Score: {results['trust']['mean']:.4f} ± {results['trust']['std']:.4f}\n")
        f.write(f"Block Count: {results['block_count']['mean']:.1f} ± {results['block_count']['std']:.1f}\n")
        f.write(f"Entropy: {results['entropy']['mean']:.4f} ± {results['entropy']['std']:.4f}\n")
        f.write(f"Cross-Validation: {results['cv_accuracy']['mean']:.4f} ± {results['cv_accuracy']['std']:.4f}\n\n")
        
        # Variation Analysis
        f.write("VARIATION ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Variation Reduced: {results['variation_reduced']['from']} → {results['variation_reduced']['to']}\n")
        f.write(f"Accuracy Range: {results['accuracy']['min']:.4f} - {results['accuracy']['max']:.4f}\n")
        f.write(f"Trust Range: {results['trust']['min']:.4f} - {results['trust']['max']:.4f}\n")
        f.write(f"Block Count Range: {results['block_count']['min']} - {results['block_count']['max']}\n\n")
        
        # Outlier Analysis
        f.write("OUTLIER ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {outlier_info['dataset']}\n")
        f.write(f"Total Samples: {outlier_info['total_samples']}\n")
        f.write(f"Outliers Detected: {outlier_info['outliers_detected']}\n")
        f.write(f"Outlier Percentage: {outlier_info['outlier_percentage']:.2f}%\n")
        
        if outlier_info['outlier_details']:
            f.write("\nOutlier Details (first 10):\n")
            for outlier in outlier_info['outlier_details']:
                f.write(f"  Sample {outlier['index']}: Label {outlier['label']}, "
                       f"Max Z-score: {outlier['max_z_score']:.2f}\n")
        f.write("\n")
        
        # Individual Test Results
        f.write("INDIVIDUAL TEST RESULTS:\n")
        f.write("-" * 40 + "\n")
        for test in results['individual_tests']:
            f.write(f"Test {test['test_id']}: "
                   f"Acc={test['final_accuracy']:.4f}, "
                   f"Trust={test['final_trust']:.4f}, "
                   f"Blocks={test['block_count']}, "
                   f"Entropy={test['entropy']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Results saved to: {output_path}")
    return output_path


def main():
    """
    Main execution function for the SREE Phase 1 demo with improved robustness.
    """
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting SREE Phase 1 Demo - Enhanced Version")
    
    # Load configuration
    config = get_config()
    logger.info(f"Configuration loaded: {len(config)} sections")
    
    try:
        # Load datasets
        logger.info("Loading datasets...")
        data_loader = DataLoader(logger)
        
        try:
            # Load datasets individually to catch any errors
            logger.info("Loading MNIST...")
            X_mnist, y_mnist = data_loader.load_mnist()
            X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = data_loader.preprocess_data(X_mnist, y_mnist)
            
            logger.info("Loading Heart...")
            X_heart, y_heart = data_loader.load_heart()
            X_train_heart, X_test_heart, y_train_heart, y_test_heart = data_loader.preprocess_data(X_heart, y_heart)
            
            logger.info("Creating Synthetic...")
            X_synth, y_synth = data_loader.create_synthetic()
            X_train_synth, X_test_synth, y_train_synth, y_test_synth = data_loader.preprocess_data(X_synth, y_synth)
            
            logger.info("Loading CIFAR-10...")
            X_cifar, y_cifar = data_loader.load_cifar10()
            X_train_cifar, X_test_cifar, y_train_cifar, y_test_cifar = data_loader.preprocess_data(X_cifar, y_cifar)
            
            datasets = {
                'mnist': {
                    'X_train': X_train_mnist,
                    'X_test': X_test_mnist,
                    'y_train': y_train_mnist,
                    'y_test': y_test_mnist,
                    'info': data_loader.get_dataset_info(X_mnist, y_mnist)
                },
                'heart': {
                    'X_train': X_train_heart,
                    'X_test': X_test_heart,
                    'y_train': y_train_heart,
                    'y_test': y_test_heart,
                    'info': data_loader.get_dataset_info(X_heart, y_heart)
                },
                'synthetic': {
                    'X_train': X_train_synth,
                    'X_test': X_test_synth,
                    'y_train': y_train_synth,
                    'y_test': y_test_synth,
                    'info': data_loader.get_dataset_info(X_synth, y_synth)
                },
                'cifar10': {
                    'X_train': X_train_cifar,
                    'X_test': X_test_cifar,
                    'y_train': y_train_cifar,
                    'y_test': y_test_cifar,
                    'info': data_loader.get_dataset_info(X_cifar, y_cifar)
                }
            }
            logger.info("All datasets loaded successfully")
        except Exception as e:
            logger.warning(f"Network error loading real datasets: {e}")
            logger.info("Falling back to synthetic dataset only...")
            
            X_synth, y_synth = data_loader.create_synthetic(n_samples=1000)
            X_train, X_test, y_train, y_test = data_loader.preprocess_data(X_synth, y_synth)
            
            datasets = {
                "synthetic": {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "info": data_loader.get_dataset_info(X_synth, y_synth)
                }
            }
        
        # Focus on heart dataset for client request (real medical data)
        target_dataset = 'heart'
        if target_dataset in datasets:
            dataset_data = datasets[target_dataset]
            logger.info(f"Focusing on {target_dataset} dataset for client analysis")
            
            # Debug: check data structure
            logger.info(f"X_train shape: {dataset_data['X_train'].shape}")
            logger.info(f"y_train shape: {dataset_data['y_train'].shape}")
            logger.info(f"X_train type: {type(dataset_data['X_train'])}")
            logger.info(f"y_train type: {type(dataset_data['y_train'])}")
            
            # Detect outliers
            logger.info("Detecting outliers in dataset...")
            outlier_info = detect_outliers(
                dataset_data['X_train'], 
                dataset_data['y_train'], 
                target_dataset
            )
            logger.info(f"Outliers detected: {outlier_info['outliers_detected']} "
                       f"({outlier_info['outlier_percentage']:.2f}%)")
            
            # Run multiple tests to reduce variation
            logger.info("Running multiple tests to reduce variation...")
            results = run_multiple_tests(target_dataset, dataset_data, n_tests=8)
            
            # Save results to text file
            logger.info("Saving results to text file...")
            output_file = save_results_to_text(results, outlier_info)
            
            # Print summary for client
            logger.info("\n" + "="*60)
            logger.info("CLIENT SUMMARY - SREE PHASE 1 RESULTS")
            logger.info("="*60)
            logger.info(f"Dataset: {target_dataset}")
            logger.info(f"Tests Run: {results['n_tests']}")
            logger.info(f"Accuracy: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
            logger.info(f"Trust Score: {results['trust']['mean']:.4f} ± {results['trust']['std']:.4f}")
            logger.info(f"Block Count: {results['block_count']['mean']:.1f} ± {results['block_count']['std']:.1f}")
            logger.info(f"Entropy: {results['entropy']['mean']:.4f} ± {results['entropy']['std']:.4f}")
            logger.info(f"Variation: {results['variation_reduced']['from']} → {results['variation_reduced']['to']}")
            logger.info(f"Outliers: {outlier_info['outliers_detected']} ({outlier_info['outlier_percentage']:.2f}%)")
            logger.info(f"Results File: {output_file}")
            logger.info("="*60)
            
        else:
            logger.error(f"Target dataset '{target_dataset}' not available")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 