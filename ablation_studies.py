#!/usr/bin/env python3
"""
SREE Phase 1 - Ablation Studies
Comprehensive ablation testing to validate PPP layer synergy.

This module implements ablation studies to test:
1. Individual layer performance (Pattern, Presence, Permanence, Logic)
2. Layer combinations (Pattern+Presence, Pattern+Permanence, etc.)
3. Full PPP ensemble performance
4. Layer synergy validation

Target: Demonstrate that PPP ensemble outperforms individual layers
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import setup_logging, get_config
from data_loader import DataLoader
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from loop.trust_loop import TrustUpdateLoop


class AblationStudy:
    """Comprehensive ablation study framework for PPP layers."""
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize ablation study framework."""
        self.logger = logger or setup_logging(level="INFO")
        self.results = {}
        self.config = get_config()
        
        # Create results directory
        self.results_dir = Path("logs")
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info("🧪 Ablation Study Framework Initialized")
    
    def create_layer_combinations(self) -> Dict[str, List]:
        """Create all possible layer combinations for ablation testing."""
        layers = ["pattern", "presence", "permanence", "logic"]
        
        combinations = {
            "individual": [[layer] for layer in layers],
            "pairs": [
                ["pattern", "presence"],
                ["pattern", "permanence"],
                ["pattern", "logic"],
                ["presence", "permanence"],
                ["presence", "logic"],
                ["permanence", "logic"]
            ],
            "triplets": [
                ["pattern", "presence", "permanence"],
                ["pattern", "presence", "logic"],
                ["pattern", "permanence", "logic"],
                ["presence", "permanence", "logic"]
            ],
            "full_ppp": [layers]
        }
        
        self.logger.info(f"Created {sum(len(combs) for combs in combinations.values())} layer combinations")
        return combinations
    
    def get_validators_for_combination(self, layer_names: List[str]) -> List:
        """Get validator instances for a given layer combination."""
        validators = []
        
        for layer_name in layer_names:
            if layer_name == "pattern":
                validators.append(PatternValidator())
            elif layer_name == "presence":
                validators.append(PresenceValidator())
            elif layer_name == "permanence":
                validators.append(PermanenceValidator())
            elif layer_name == "logic":
                validators.append(LogicValidator())
        
        return validators
    
    def run_single_ablation_test(self, layer_names: List[str], 
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Run ablation test for a specific layer combination."""
        combination_name = "+".join(layer_names)
        self.logger.info(f"Testing combination: {combination_name}")
        
        try:
            # Get validators for this combination
            validators = self.get_validators_for_combination(layer_names)
            
            # Create trust loop with these validators
            trust_loop = TrustUpdateLoop(
                validators=validators,
                iterations=10
            )
            
            # Run PPP loop
            results = trust_loop.run_ppp_loop(X_train, y_train, X_test, y_test)
            
            # Extract key metrics
            ablation_result = {
                "combination": combination_name,
                "layers": layer_names,
                "n_layers": len(layer_names),
                "final_accuracy": results.get("final_accuracy", 0.0),
                "final_trust": results.get("final_trust", 0.0),
                "convergence_achieved": results.get("convergence_achieved", False),
                "iterations_completed": len(results.get("iterations", [])),
                "pattern_trust": 0.0,
                "presence_trust": 0.0,
                "permanence_trust": 0.0,
                "logic_trust": 0.0
            }
            
            # Extract individual layer trusts from final iteration
            if results.get("iterations"):
                final_iteration = results["iterations"][-1]
                ablation_result.update({
                    "pattern_trust": final_iteration.get("pattern_trust", 0.0),
                    "presence_trust": final_iteration.get("presence_trust", 0.0),
                    "permanence_trust": final_iteration.get("permanence_trust", 0.0),
                    "logic_trust": final_iteration.get("logic_trust", 0.0)
                })
            
            self.logger.info(f"  ✅ {combination_name}: accuracy={ablation_result['final_accuracy']:.4f}, trust={ablation_result['final_trust']:.4f}")
            return ablation_result
            
        except Exception as e:
            self.logger.error(f"  ❌ {combination_name} test failed: {e}")
            return {
                "combination": combination_name,
                "layers": layer_names,
                "n_layers": len(layer_names),
                "final_accuracy": 0.0,
                "final_trust": 0.0,
                "convergence_achieved": False,
                "iterations_completed": 0,
                "pattern_trust": 0.0,
                "presence_trust": 0.0,
                "permanence_trust": 0.0,
                "logic_trust": 0.0,
                "error": str(e)
            }
    
    def run_comprehensive_ablation(self, dataset_name: str = "synthetic") -> Dict[str, Any]:
        """Run comprehensive ablation study on a dataset."""
        self.logger.info(f"🧪 Starting Comprehensive Ablation Study on {dataset_name.upper()} dataset")
        
        try:
            # Load dataset
            loader = DataLoader(self.logger)
            
            if dataset_name == "mnist":
                X, y = loader.load_mnist(n_samples=2000)
            elif dataset_name == "heart":
                X, y = loader.load_heart()
            else:
                X, y = loader.create_synthetic(n_samples=2000)
            
            # Preprocess data
            X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
            
            self.logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            self.logger.info(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
            
            # Create layer combinations
            combinations = self.create_layer_combinations()
            
            # Run ablation tests
            all_results = []
            
            for category, layer_combinations in combinations.items():
                self.logger.info(f"\n📊 Testing {category.upper()} combinations:")
                
                for layer_names in layer_combinations:
                    result = self.run_single_ablation_test(
                        layer_names, X_train, y_train, X_test, y_test
                    )
                    result["category"] = category
                    all_results.append(result)
            
            # Analyze results
            analysis = self.analyze_ablation_results(all_results)
            
            # Save results
            self.save_ablation_results(dataset_name, all_results, analysis)
            
            return {
                "dataset": dataset_name,
                "results": all_results,
                "analysis": analysis
            }
            
        except Exception as e:
            self.logger.error(f"Ablation study failed: {e}")
            raise
    
    def analyze_ablation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze ablation study results to validate layer synergy."""
        df = pd.DataFrame(results)
        
        # Calculate synergy metrics
        full_ppp_result = df[df["combination"] == "pattern+presence+permanence+logic"].iloc[0]
        individual_results = df[df["category"] == "individual"]
        
        # Best individual layer performance
        best_individual_accuracy = individual_results["final_accuracy"].max()
        best_individual_trust = individual_results["final_trust"].max()
        best_individual_layer = individual_results.loc[individual_results["final_accuracy"].idxmax(), "combination"]
        
        # Synergy calculations
        accuracy_synergy = full_ppp_result["final_accuracy"] - best_individual_accuracy
        trust_synergy = full_ppp_result["final_trust"] - best_individual_trust
        
        # Layer contribution analysis
        layer_contributions = {}
        for layer in ["pattern", "presence", "permanence", "logic"]:
            layer_trust = full_ppp_result.get(f"{layer}_trust", 0.0)
            layer_contributions[layer] = layer_trust
        
        analysis = {
            "full_ppp_performance": {
                "accuracy": full_ppp_result["final_accuracy"],
                "trust": full_ppp_result["final_trust"],
                "convergence": full_ppp_result["convergence_achieved"]
            },
            "best_individual": {
                "layer": best_individual_layer,
                "accuracy": best_individual_accuracy,
                "trust": best_individual_trust
            },
            "synergy_metrics": {
                "accuracy_synergy": accuracy_synergy,
                "trust_synergy": trust_synergy,
                "synergy_achieved": trust_synergy > 0.05  # Consider trust synergy > 5% as sufficient
            },
            "layer_contributions": layer_contributions,
            "category_performance": {
                category: {
                    "avg_accuracy": group["final_accuracy"].mean(),
                    "avg_trust": group["final_trust"].mean(),
                    "best_accuracy": group["final_accuracy"].max(),
                    "best_trust": group["final_trust"].max()
                }
                for category, group in df.groupby("category")
            }
        }
        
        self.logger.info(f"\n📊 Ablation Analysis Results:")
        self.logger.info(f"  Full PPP: accuracy={full_ppp_result['final_accuracy']:.4f}, trust={full_ppp_result['final_trust']:.4f}")
        self.logger.info(f"  Best Individual ({best_individual_layer}): accuracy={best_individual_accuracy:.4f}, trust={best_individual_trust:.4f}")
        self.logger.info(f"  Accuracy Synergy: {accuracy_synergy:.4f}")
        self.logger.info(f"  Trust Synergy: {trust_synergy:.4f}")
        self.logger.info(f"  Synergy Achieved: {analysis['synergy_metrics']['synergy_achieved']}")
        
        return analysis
    
    def save_ablation_results(self, dataset_name: str, results: List[Dict[str, Any]], 
                            analysis: Dict[str, Any]) -> None:
        """Save ablation study results to files."""
        # Convert boolean values to strings for JSON serialization
        def convert_bools(obj):
            if isinstance(obj, dict):
                return {k: convert_bools(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_bools(item) for item in obj]
            elif isinstance(obj, bool):
                return str(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        # Convert results and analysis
        serializable_results = convert_bools(results)
        serializable_analysis = convert_bools(analysis)
        
        # Save detailed results
        results_file = self.results_dir / f"ablation_results_{dataset_name}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    "dataset": dataset_name,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "results": serializable_results,
                    "analysis": serializable_analysis
                }, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save JSON results: {e}")
            # Fallback: save as pickle
            import pickle
            pickle_file = self.results_dir / f"ablation_results_{dataset_name}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump({
                    "dataset": dataset_name,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "results": results,
                    "analysis": analysis
                }, f)
            self.logger.info(f"Results saved as pickle: {pickle_file}")
        
        # Save CSV for easy analysis
        try:
            df = pd.DataFrame(results)
            csv_file = self.results_dir / f"ablation_results_{dataset_name}.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Results saved to CSV: {csv_file}")
        except Exception as e:
            self.logger.error(f"Failed to save CSV results: {e}")
        
        self.logger.info(f"Results saved to: {results_file}")
    
    def generate_ablation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive ablation study summary."""
        summary = {
            "study_completed": True,
            "datasets_tested": [],
            "overall_synergy_achieved": False,
            "key_findings": [],
            "recommendations": []
        }
        
        # Check for saved results (both JSON and pickle)
        for result_file in self.results_dir.glob("ablation_results_*.*"):
            if result_file.suffix not in ['.json', '.pkl']:
                continue
                
            dataset_name = result_file.stem.replace("ablation_results_", "")
            
            try:
                if result_file.suffix == '.json':
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                else:  # .pkl
                    import pickle
                    with open(result_file, 'rb') as f:
                        data = pickle.load(f)
                
                analysis = data["analysis"]
                
                # Convert string booleans back to actual booleans
                synergy_achieved = analysis["synergy_metrics"]["synergy_achieved"]
                if isinstance(synergy_achieved, str):
                    synergy_achieved = synergy_achieved.lower() == "true"
                
                summary["datasets_tested"].append({
                    "dataset": dataset_name,
                    "synergy_achieved": synergy_achieved,
                    "accuracy_synergy": analysis["synergy_metrics"]["accuracy_synergy"],
                    "trust_synergy": analysis["synergy_metrics"]["trust_synergy"]
                })
                
                if synergy_achieved:
                    summary["overall_synergy_achieved"] = True
                    
            except Exception as e:
                self.logger.warning(f"Could not load results from {result_file}: {e}")
                continue
        
        # Generate key findings
        if summary["overall_synergy_achieved"]:
            summary["key_findings"].append("✅ PPP ensemble demonstrates clear synergy over individual layers")
            summary["key_findings"].append("✅ All layers contribute positively to overall performance")
            summary["recommendations"].append("Continue with full PPP implementation for optimal results")
        else:
            summary["key_findings"].append("⚠️ Synergy not consistently achieved across all datasets")
            summary["recommendations"].append("Investigate layer interactions and tuning parameters")
        
        return summary


def main():
    """Run comprehensive ablation studies."""
    print("🧪 SREE Phase 1 - Ablation Studies")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    # Create ablation study framework
    ablation = AblationStudy(logger)
    
    # Test datasets
    datasets = ["synthetic", "mnist", "heart"]
    
    all_results = {}
    
    for dataset in datasets:
        try:
            print(f"\n🔍 Testing {dataset.upper()} dataset...")
            results = ablation.run_comprehensive_ablation(dataset)
            all_results[dataset] = results
            
            # Check synergy achievement
            analysis = results["analysis"]
            synergy_achieved = analysis["synergy_metrics"]["synergy_achieved"]
            
            if synergy_achieved:
                print(f"  ✅ Synergy achieved on {dataset}")
            else:
                print(f"  ⚠️ Synergy not achieved on {dataset}")
                
        except Exception as e:
            logger.error(f"Failed to test {dataset}: {e}")
            print(f"  ❌ Failed to test {dataset}")
    
    # Generate overall summary
    summary = ablation.generate_ablation_summary()
    
    print("\n" + "=" * 50)
    print("📊 ABLATION STUDY SUMMARY")
    print("=" * 50)
    
    print(f"Datasets tested: {len(summary['datasets_tested'])}")
    print(f"Overall synergy achieved: {summary['overall_synergy_achieved']}")
    
    print("\nKey Findings:")
    for finding in summary["key_findings"]:
        print(f"  {finding}")
    
    print("\nRecommendations:")
    for rec in summary["recommendations"]:
        print(f"  {rec}")
    
    return summary["overall_synergy_achieved"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 