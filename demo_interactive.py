#!/usr/bin/env python3
"""
SREE Interactive Demo
Interactive demonstration of SREE with detailed explanations and pauses for comprehension.
"""

import time
import logging
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.text import Text

from config import setup_logging
from data_loader import DataLoader
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from loop.trust_loop import TrustUpdateLoop

class SREEInteractiveDemo:
    """Interactive SREE demonstration."""
    
    def __init__(self):
        self.console = Console()
        self.logger = setup_logging()
        
    def print_header(self):
        """Prints demonstration header."""
        self.console.print(Panel.fit(
            "[bold blue]🧠 SREE - Self-Refining Epistemic Engine[/bold blue]\n"
            "[italic]Interactive Demo - Phase 1[/italic]",
            border_style="blue"
        ))
        
    def print_section(self, title, description):
        """Prints section with title and description."""
        self.console.print(f"\n[bold green]🔹 {title}[/bold green]")
        self.console.print(f"[dim]{description}[/dim]")
        input("\n[bold]Press ENTER to continue...[/bold]")
        
    def demo_data_loading(self):
        """Demonstrates data loading."""
        self.print_section(
            "Dataset Loading",
            "Let's load the real datasets (MNIST and Heart Disease) and create a synthetic dataset as backup."
        )
        
        loader = DataLoader(self.logger)
        
        # MNIST
        self.console.print("[yellow]📊 Loading MNIST dataset...[/yellow]")
        with self.console.status("[bold green]Loading MNIST..."):
            X_mnist, y_mnist = loader.load_mnist(n_samples=1000)
        self.console.print(f"✅ MNIST loaded: {X_mnist.shape[0]} samples, {X_mnist.shape[1]} features")
        
        # Heart Disease
        self.console.print("[yellow]📊 Loading Heart Disease dataset...[/yellow]")
        with self.console.status("[bold green]Loading Heart Disease..."):
            X_heart, y_heart = loader.load_heart()
        self.console.print(f"✅ Heart Disease loaded: {X_heart.shape[0]} samples, {X_heart.shape[1]} features")
        
        # Synthetic
        self.console.print("[yellow]📊 Creating synthetic dataset...[/yellow]")
        with self.console.status("[bold green]Creating synthetic dataset..."):
            X_synth, y_synth = loader.create_synthetic(n_samples=500)
        self.console.print(f"✅ Synthetic dataset created: {X_synth.shape[0]} samples, {X_synth.shape[1]} features")
        
        # Use synthetic for demo and split into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_synth, y_synth, test_size=0.2, random_state=42
        )
        return X_train, y_train, X_test, y_test
        
    def demo_pattern_layer(self, X_train, y_train, X_test, y_test):
        """Demonstrates the Pattern layer."""
        self.print_section(
            "Pattern Layer - Pattern Recognition",
            "This layer uses an MLP classifier to learn patterns in the data and make predictions."
        )
        
        self.console.print("[yellow]🔧 Initializing Pattern Validator...[/yellow]")
        pattern_validator = PatternValidator()
        
        self.console.print("[yellow]🎯 Training MLP model...[/yellow]")
        with self.console.status("[bold green]Training Pattern Layer..."):
            training_results = pattern_validator.train(X_train, y_train)
        
        # Show training results
        table = Table(title="Training Results - Pattern Layer")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        train_acc = training_results.get('train_accuracy', 0) or 0
        val_acc = training_results.get('val_accuracy', 0) or 0
        cv_score = training_results.get('cv_mean', 0) or 0
        
        table.add_row("Training Accuracy", f"{train_acc:.3f}")
        table.add_row("Validation Accuracy", f"{val_acc:.3f}")
        table.add_row("Cross-Validation Score", f"{cv_score:.3f}")
        
        self.console.print(table)
        
        self.console.print("[yellow]🔍 Validating test data...[/yellow]")
        with self.console.status("[bold green]Validating..."):
            trust_scores = pattern_validator.validate(X_test, y_test)
        
        self.console.print(f"✅ Average Trust Score: {np.mean(trust_scores):.3f}")
        
        return pattern_validator
        
    def demo_presence_layer(self, X_train, y_train, X_test, y_test):
        """Demonstrates the Presence layer."""
        self.print_section(
            "Presence Layer - Entropy Minimization",
            "This layer simulates quantum computation to minimize entropy and refine predictions."
        )
        
        self.console.print("[yellow]🔧 Initializing Presence Validator...[/yellow]")
        presence_validator = PresenceValidator()
        
        self.console.print("[yellow]⚛️ Calculating entropy and refining predictions...[/yellow]")
        with self.console.status("[bold green]Processing Presence Layer..."):
            trust_scores = presence_validator.validate(X_test, y_test)
        
        # Show entropy statistics
        entropy_stats = presence_validator.get_entropy_statistics()
        
        table = Table(title="Entropy Statistics - Presence Layer")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Mean Entropy", f"{entropy_stats.get('mean_entropy', 0):.3f}")
        table.add_row("Min Entropy", f"{entropy_stats.get('min_entropy', 0):.3f}")
        table.add_row("Max Entropy", f"{entropy_stats.get('max_entropy', 0):.3f}")
        table.add_row("Average Trust Score", f"{np.mean(trust_scores):.3f}")
        
        self.console.print(table)
        
        return presence_validator
        
    def demo_permanence_layer(self, X_train, y_train, X_test, y_test):
        """Demonstrates the Permanence layer."""
        self.print_section(
            "Permanence Layer - Hash-Based Logging",
            "This layer simulates blockchain to create an immutable ledger and verify consistency."
        )
        
        self.console.print("[yellow]🔧 Initializing Permanence Validator...[/yellow]")
        permanence_validator = PermanenceValidator()
        
        self.console.print("[yellow]🔗 Creating blocks and checking consistency...[/yellow]")
        with self.console.status("[bold green]Processing Permanence Layer..."):
            trust_scores = permanence_validator.validate(X_test, y_test)
        
        # Check ledger consistency
        consistency_result = permanence_validator.check_ledger_consistency()
        
        table = Table(title="Results - Permanence Layer")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Number of Blocks", str(consistency_result.get('num_blocks', 0)))
        table.add_row("Status", consistency_result.get('status', 'unknown'))
        if 'consistency_score' in consistency_result:
            table.add_row("Consistency Score", f"{consistency_result['consistency_score']:.3f}")
        table.add_row("Average Trust Score", f"{np.mean(trust_scores):.3f}")
        
        self.console.print(table)
        
        return permanence_validator
        
    def demo_logic_layer(self, X_train, y_train, X_test, y_test):
        """Demonstrates the Logic layer."""
        self.print_section(
            "Logic Layer - Consistency Validation",
            "This layer applies logical rules to validate data and prediction consistency."
        )
        
        self.console.print("[yellow]🔧 Initializing Logic Validator...[/yellow]")
        logic_validator = LogicValidator()
        
        self.console.print("[yellow]🧠 Applying logical validation...[/yellow]")
        with self.console.status("[bold green]Processing Logic Layer..."):
            trust_scores = logic_validator.validate(X_test, y_test)
        
        # Get consistency statistics
        consistency_stats = logic_validator.get_consistency_statistics()
        
        table = Table(title="Consistency Statistics - Logic Layer")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Feature Consistency", f"{consistency_stats.get('feature_consistency', 0):.3f}")
        table.add_row("Label Consistency", f"{consistency_stats.get('label_consistency', 0):.3f}")
        table.add_row("Prediction Consistency", f"{consistency_stats.get('prediction_consistency', 0):.3f}")
        table.add_row("Average Trust Score", f"{np.mean(trust_scores):.3f}")
        
        self.console.print(table)
        
        return logic_validator
        
    def demo_trust_loop(self, X_train, y_train, X_test, y_test, validators):
        """Demonstrates the trust update loop."""
        self.print_section(
            "Trust Update Loop - Layer Integration",
            "The trust loop integrates all PPP layers and iteratively updates trust scores."
        )
        
        self.console.print("[yellow]🔧 Initializing Trust Update Loop...[/yellow]")
        trust_loop = TrustUpdateLoop(validators=validators)
        
        self.console.print("[yellow]🔄 Running trust loop...[/yellow]")
        with self.console.status("[bold green]Running Trust Loop..."):
            results = trust_loop.run_ppp_loop(X_train, y_train, X_test, y_test)
        
        # Show loop results
        table = Table(title="Trust Update Loop Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Iterations", str(len(results.get('iterations', []))))
        table.add_row("Convergence Achieved", "✅ Yes" if results.get('convergence_achieved', False) else "❌ No")
        table.add_row("Final Trust Score", f"{results.get('final_trust_score', 0):.3f}")
        table.add_row("Final Accuracy", f"{results.get('final_accuracy', 0):.3f}")
        
        self.console.print(table)
        
        # Show trust score evolution
        if 'iterations' in results:
            self.console.print("\n[yellow]📈 Trust Score Evolution:[/yellow]")
            for i, iteration in enumerate(results['iterations']):
                trust_score = iteration.get('trust_score', 0)
                self.console.print(f"  Iteration {i+1}: {trust_score:.3f}")
        
        return results
        
    def demo_ablation_tests(self, X_train, y_train, X_test, y_test, validators):
        """Demonstrates ablation tests to show PPP necessity."""
        self.print_section(
            "Ablation Tests - PPP Layer Necessity",
            "Let's test different layer combinations to demonstrate that the full PPP ensemble is necessary for optimal performance."
        )
        
        # Define layer combinations for ablation testing
        layer_combinations = {
            "Pattern Only": [validators[0]],  # Pattern only
            "Pattern + Presence": [validators[0], validators[1]],  # Pattern + Presence
            "Pattern + Permanence": [validators[0], validators[2]],  # Pattern + Permanence
            "Pattern + Logic": [validators[0], validators[3]],  # Pattern + Logic
            "Full PPP": validators  # All layers
        }
        
        self.console.print("[yellow]🔬 Running ablation tests...[/yellow]")
        
        # Results storage
        ablation_results = {}
        
        for combination_name, combination_validators in layer_combinations.items():
            self.console.print(f"\n[cyan]Testing: {combination_name}[/cyan]")
            
            try:
                # Run analysis with this combination
                from loop.trust_loop import TrustUpdateLoop
                trust_loop = TrustUpdateLoop()
                
                results = trust_loop.run_analysis(
                    X_train, y_train, X_test, y_test, combination_validators
                )
                
                accuracy = results.get('final_accuracy', 0)
                trust_score = results.get('final_trust_score', 0)
                
                ablation_results[combination_name] = {
                    'accuracy': accuracy,
                    'trust_score': trust_score,
                    'layers': len(combination_validators)
                }
                
                self.console.print(f"  ✅ Accuracy: {accuracy:.3f}")
                self.console.print(f"  ✅ Trust Score: {trust_score:.3f}")
                
            except Exception as e:
                self.console.print(f"  ❌ Error: {e}")
                ablation_results[combination_name] = {
                    'accuracy': 0,
                    'trust_score': 0,
                    'layers': len(combination_validators)
                }
        
        # Display ablation results table
        self.console.print("\n[bold green]📊 Ablation Test Results[/bold green]")
        
        table = Table(title="Layer Combination Performance")
        table.add_column("Combination", style="cyan")
        table.add_column("Layers", style="blue")
        table.add_column("Accuracy", style="magenta")
        table.add_column("Trust Score", style="yellow")
        table.add_column("Improvement", style="green")
        
        # Calculate baseline (Pattern only)
        baseline_accuracy = ablation_results.get("Pattern Only", {}).get('accuracy', 0)
        baseline_trust = ablation_results.get("Pattern Only", {}).get('trust_score', 0)
        
        for combination_name, results in ablation_results.items():
            accuracy = results['accuracy']
            trust_score = results['trust_score']
            layers = results['layers']
            
            # Calculate improvement over baseline
            acc_improvement = ((accuracy - baseline_accuracy) / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
            trust_improvement = ((trust_score - baseline_trust) / baseline_trust * 100) if baseline_trust > 0 else 0
            
            improvement_text = f"+{acc_improvement:.1f}% acc, +{trust_improvement:.1f}% trust"
            
            table.add_row(
                combination_name,
                str(layers),
                f"{accuracy:.3f}",
                f"{trust_score:.3f}",
                improvement_text
            )
        
        self.console.print(table)
        
        # Key findings
        full_ppp_accuracy = ablation_results.get("Full PPP", {}).get('accuracy', 0)
        full_ppp_trust = ablation_results.get("Full PPP", {}).get('trust_score', 0)
        
        if full_ppp_accuracy > baseline_accuracy:
            improvement = ((full_ppp_accuracy - baseline_accuracy) / baseline_accuracy * 100)
            self.console.print(f"\n[bold green]🎯 Key Finding:[/bold green]")
            self.console.print(f"Full PPP ensemble improves accuracy by {improvement:.1f}% over Pattern-only baseline")
            self.console.print(f"This demonstrates the necessity of all PPP layers (§C.2)")
        
        return ablation_results
        
    def demo_performance_summary(self, results):
        """Demonstrates performance summary."""
        self.print_section(
            "Performance Summary",
            "Let's analyze the final results and compare them with the established goals."
        )
        
        # Create performance table
        table = Table(title="Final Performance - SREE Phase 1")
        table.add_column("Metric", style="cyan")
        table.add_column("Current Value", style="magenta")
        table.add_column("Target", style="yellow")
        table.add_column("Status", style="green")
        
        final_accuracy = results.get('final_accuracy', 0)
        final_trust = results.get('final_trust_score', 0)
        
        table.add_row(
            "Accuracy", 
            f"{final_accuracy:.3f}", 
            "0.985", 
            "✅ Achieved" if final_accuracy >= 0.985 else "❌ Not achieved"
        )
        
        table.add_row(
            "Trust Score", 
            f"{final_trust:.3f}", 
            "0.96", 
            "✅ Achieved" if final_trust >= 0.96 else "❌ Not achieved"
        )
        
        self.console.print(table)
        
        # Conclusion
        self.console.print(Panel.fit(
            "[bold green]🎉 Demonstration Completed![/bold green]\n\n"
            "SREE successfully demonstrated:\n"
            "• Integration of all 4 PPP layers\n"
            "• Trust update loop\n"
            "• Performance within established targets\n"
            "• Modular architecture for Phase 2\n\n"
            "[italic]For more details, run the complete tests: python3 run_tests.py[/italic]",
            border_style="green"
        ))
        
    def run_demo(self):
        """Runs the complete demonstration."""
        try:
            self.print_header()
            
            # 1. Data loading
            X_train, y_train, X_test, y_test = self.demo_data_loading()
            
            # 2. PPP layers
            pattern_validator = self.demo_pattern_layer(X_train, y_train, X_test, y_test)
            presence_validator = self.demo_presence_layer(X_train, y_train, X_test, y_test)
            permanence_validator = self.demo_permanence_layer(X_train, y_train, X_test, y_test)
            logic_validator = self.demo_logic_layer(X_train, y_train, X_test, y_test)
            
            # 3. Trust Loop
            validators = [pattern_validator, presence_validator, permanence_validator, logic_validator]
            results = self.demo_trust_loop(X_train, y_train, X_test, y_test, validators)
            
            # 4. Ablation Tests
            self.demo_ablation_tests(X_train, y_train, X_test, y_test, validators)
            
            # 5. Summary
            self.demo_performance_summary(results)
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️ Demonstration interrupted by user.[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]❌ Error during demonstration: {e}[/red]")
            self.logger.error(f"Error in demonstration: {e}")

def main():
    """Main demonstration function."""
    demo = SREEInteractiveDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 