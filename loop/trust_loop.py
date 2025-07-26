"""
SREE Phase 1 Demo - Trust Update Loop
Recursive trust update mechanism for PPP convergence.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
from datetime import datetime

from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from config import PPP_CONFIG, LOGS_DIR


class TrustUpdateLoop:
    """
    Trust Update Loop - Recursive trust update mechanism for PPP convergence.
    
    This class implements the core trust update mechanism that coordinates all
    PPP layers to achieve convergence of trust scores. It manages the iterative
    process of updating trust scores based on the outputs of all validators.
    
    Target: Achieve trust score convergence T ≈ 0.96 with ~98.5% accuracy
    """
    
    def __init__(self, name: str = "TrustUpdateLoop", validators: List = None, **kwargs):
        """
        Initialize Trust Update Loop with PPP validators.
        
        Args:
            name: Loop name
            validators: List of validator instances to use (if None, uses all default validators)
            **kwargs: Additional configuration parameters
        """
        # Initialize state attributes first
        self._iterations = PPP_CONFIG["iterations"]
        self._gamma = PPP_CONFIG["gamma"]  # State update rate
        self._alpha = PPP_CONFIG["alpha"]  # Trust update rate
        self._beta = PPP_CONFIG["beta"]    # Permanence weight
        self._delta = PPP_CONFIG["delta"]  # Logic weight
        self._initial_trust = PPP_CONFIG["initial_trust"]
        self._initial_state = PPP_CONFIG["initial_state"]
        
        # Override with recommended values for optimal performance
        self._alpha = 0.7  # Trust update rate
        self._beta = 0.6   # Permanence weight
        self._gamma = 0.1  # State update rate
        self._delta = 0.1  # Logic weight
        
        # Initialize validators
        if validators is None:
            # Use default validators
            self._pattern_validator = PatternValidator()
            self._presence_validator = PresenceValidator()
            self._permanence_validator = PermanenceValidator()
            self._logic_validator = LogicValidator()
            self._validators = [self._pattern_validator, self._presence_validator, 
                              self._permanence_validator, self._logic_validator]
        else:
            # Use custom validators
            self._validators = validators
            # Map validators by type for backward compatibility
            self._pattern_validator = None
            self._presence_validator = None
            self._permanence_validator = None
            self._logic_validator = None
            
            for validator in validators:
                if isinstance(validator, PatternValidator):
                    self._pattern_validator = validator
                elif isinstance(validator, PresenceValidator):
                    self._presence_validator = validator
                elif isinstance(validator, PermanenceValidator):
                    self._permanence_validator = validator
                elif isinstance(validator, LogicValidator):
                    self._logic_validator = validator
        
        # Trust update history
        self._trust_history = []
        self._accuracy_history = []
        self._convergence_history = []
        
        # Current state
        self._current_trust = self._initial_trust
        self._current_state = self._initial_state
        
        # Previous values for recursive formulas
        self._previous_trust = None
        self._previous_state = None
        
        # Get loop configuration
        loop_config = PPP_CONFIG.copy()
        loop_config.update(kwargs)
        
        # Store configuration
        self._config = loop_config
        
        # Call parent constructor last
        super().__init__()
    
    def run_ppp_loop(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Run the complete PPP loop with trust updates.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Results dictionary with final metrics
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting PPP loop with {self._iterations} iterations")
        
        # Train Pattern validator if available
        if self._pattern_validator is not None:
            logger.info("Training Pattern validator...")
            self._pattern_validator.train(X_train, y_train)
        
        # Initialize results
        results = {
            "iterations": [],
            "final_accuracy": 0.0,
            "final_trust": 0.0,
            "convergence_achieved": False,
            "convergence_iteration": -1
        }
        
        # Run iterations
        for iteration in range(self._iterations):
            logger.info(f"PPP Iteration {iteration + 1}/{self._iterations}")
            
            # Step 1: Pattern Layer
            if self._pattern_validator is not None:
                pattern_trust = self._pattern_validator.validate(X_test, y_test)
                pattern_predictions = self._pattern_validator.predictions
                pattern_probabilities = self._pattern_validator.probabilities
            else:
                # Use dummy values if pattern validator not available
                pattern_trust = np.ones(len(y_test)) * 0.5
                pattern_predictions = np.random.randint(0, 2, len(y_test))
                pattern_probabilities = np.random.rand(len(y_test), 2)
            
            # Step 2: Presence Layer (quantum validation)
            if self._presence_validator is not None:
                # Calculate quantum validation V_q
                v_q = self._presence_validator.calculate_quantum_validation(pattern_probabilities)
                
                # Adjust probabilities with quantum validation
                adjusted_probabilities = self._presence_validator.adjust_probabilities_with_quantum_validation(pattern_probabilities)
                
                # Refine predictions
                refined_predictions, refined_probabilities = self._presence_validator.refine_predictions(
                    pattern_predictions, adjusted_probabilities, X_test
                )
                presence_trust = self._presence_validator.validate(X_test, y_test)
            else:
                # Use pattern outputs if presence validator not available
                refined_predictions = pattern_predictions
                refined_probabilities = pattern_probabilities
                presence_trust = np.ones(len(y_test)) * 0.5
                v_q = np.ones(len(y_test)) * 0.5
            
            # Step 3: Permanence Layer (blockchain validation)
            if self._permanence_validator is not None:
                # Calculate blockchain validation V_b
                v_b = self._permanence_validator.calculate_blockchain_validation(y_test, refined_probabilities)
                permanence_trust = self._permanence_validator.validate(X_test, y_test)
            else:
                permanence_trust = np.ones(len(y_test)) * 0.5
                v_b = np.ones(len(y_test)) * 0.5
            
            # Step 4: Logic Layer (symbolic validation)
            if self._logic_validator is not None:
                # Calculate symbolic validation V_l
                v_l = self._logic_validator.calculate_symbolic_validation(refined_predictions, X_test)
                logic_trust = self._logic_validator.validate(X_test, y_test)
            else:
                logic_trust = np.ones(len(y_test)) * 0.5
                v_l = np.ones(len(y_test)) * 0.5
            
            # Step 5: Apply recursive trust update formulas
            updated_trust, updated_state = self._apply_recursive_trust_formulas(v_q, v_b, v_l)
            
            # Step 6: Calculate accuracy
            # Ensure predictions and y_test are properly formatted
            final_predictions = np.array(refined_predictions).astype(int)
            y_test_formatted = np.array(y_test).astype(int)
            accuracy = np.mean(final_predictions == y_test_formatted)
            
            # Step 7: Check convergence
            convergence = self._check_convergence(updated_trust, accuracy)
            
            # Store iteration results
            iteration_result = {
                "iteration": iteration + 1,
                "pattern_trust": float(np.mean(pattern_trust)),
                "presence_trust": float(np.mean(presence_trust)),
                "permanence_trust": float(np.mean(permanence_trust)),
                "logic_trust": float(np.mean(logic_trust)),
                "v_q": float(np.mean(v_q)),
                "v_b": float(np.mean(v_b)),
                "v_l": float(np.mean(v_l)),
                "updated_trust": float(np.mean(updated_trust)),
                "updated_state": float(np.mean(updated_state)),
                "accuracy": float(accuracy),
                "convergence": convergence
            }
            
            results["iterations"].append(iteration_result)
            
            # Store history
            self._trust_history.append({
                "mean_trust": float(np.mean(updated_trust)),
                "std_trust": float(np.std(updated_trust)),
                "iteration": iteration
            })
            self._accuracy_history.append(float(accuracy))
            self._convergence_history.append(convergence)
            
            # Update current state
            self._current_trust = float(np.mean(updated_trust))
            self._current_state = float(accuracy)
            
            # Log progress
            logger.info(f"  Accuracy: {accuracy:.4f}, Trust: {np.mean(updated_trust):.4f}, "
                       f"Convergence: {convergence}")
            
            # Check for early convergence
            if convergence and iteration > 2:
                logger.info(f"Convergence achieved at iteration {iteration + 1}")
                results["convergence_achieved"] = True
                results["convergence_iteration"] = iteration + 1
                break
        
        # Set final results
        if results["iterations"]:
            final_iteration = results["iterations"][-1]
            results["final_accuracy"] = final_iteration["accuracy"]
            results["final_trust"] = final_iteration["updated_trust"]
        
        logger.info(f"PPP loop completed. Final accuracy: {results['final_accuracy']:.4f}, "
                   f"Final trust: {results['final_trust']:.4f}")
        
        return results
    
    def _apply_recursive_trust_formulas(self, v_q: np.ndarray, v_b: np.ndarray, v_l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply recursive trust update formulas.
        
        Args:
            v_q: Quantum validation scores from Presence layer
            v_b: Blockchain validation scores from Permanence layer
            v_l: Symbolic validation scores from Logic layer
            
        Returns:
            Tuple of (updated_trust, updated_state)
        """
        # Formula 1: V_t = β * V_b + (1 - β - δ) * V_q + δ * V_l
        v_t = (self._beta * v_b + 
               (1 - self._beta - self._delta) * v_q + 
               self._delta * v_l)
        
        # Formula 2: S_t = S_prev + γ * (V_t - S_prev)
        if self._previous_state is not None:
            s_t = self._previous_state + self._gamma * (v_t - self._previous_state)
        else:
            s_t = v_t  # First iteration
        
        # Formula 3: T_t = α * V_t + (1 - α) * T_prev
        if self._previous_trust is not None:
            t_t = self._alpha * v_t + (1 - self._alpha) * self._previous_trust
        else:
            t_t = v_t  # First iteration
        
        # Store current values for next iteration
        self._previous_trust = t_t.copy()
        self._previous_state = s_t.copy()
        
        return t_t, s_t
    
    def run_analysis(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    validators: List = None) -> Dict[str, Any]:
        """
        Run analysis with custom validators (for ablation testing).
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            validators: List of validators to use (if None, uses default validators)
            
        Returns:
            Results dictionary with final metrics
        """
        # Create temporary trust loop with custom validators
        if validators is not None:
            temp_loop = TrustUpdateLoop(validators=validators)
            return temp_loop.run_ppp_loop(X_train, y_train, X_test, y_test)
        else:
            return self.run_ppp_loop(X_train, y_train, X_test, y_test)
    
    def _update_trust_scores(self, pattern_trust: np.ndarray, presence_trust: np.ndarray,
                            permanence_trust: np.ndarray, logic_trust: np.ndarray,
                            final_trust: np.ndarray) -> np.ndarray:
        """
        Update trust scores using weighted combination of all validators.
        
        Args:
            pattern_trust: Trust scores from Pattern validator
            presence_trust: Trust scores from Presence validator
            permanence_trust: Trust scores from Permanence validator
            logic_trust: Trust scores from Logic validator
            final_trust: Current final trust scores
            
        Returns:
            Updated trust scores
        """
        logger = logging.getLogger(__name__)
        
        # Enhanced trust score combination with adaptive weights
        iteration = len(self._trust_history)
        
        # Adaptive weights based on iteration progress
        if iteration < 5:
            # Early iterations: focus on pattern and presence
            pattern_weight = 0.4
            presence_weight = 0.3
            permanence_weight = 0.2
            logic_weight = 0.1
        elif iteration < 10:
            # Middle iterations: balance all components
            pattern_weight = 0.3
            presence_weight = 0.3
            permanence_weight = 0.25
            logic_weight = 0.15
        else:
            # Later iterations: emphasize permanence and logic for stability
            pattern_weight = 0.25
            presence_weight = 0.25
            permanence_weight = 0.3
            logic_weight = 0.2
        
        # Calculate weighted trust scores
        weighted_trust = (pattern_weight * pattern_trust + 
                         presence_weight * presence_trust + 
                         permanence_weight * permanence_trust + 
                         logic_weight * logic_trust)
        
        # Apply trust score enhancement based on consistency
        consistency_boost = np.minimum(0.1, np.std([pattern_trust, presence_trust, 
                                                   permanence_trust, logic_trust], axis=0))
        
        # Boost trust scores for consistent predictions
        enhanced_trust = np.minimum(1.0, weighted_trust + consistency_boost)
        
        # Apply iterative improvement
        if iteration > 0:
            # Use exponential moving average for stability
            alpha = min(0.3, 0.1 + iteration * 0.02)  # Increasing weight over iterations
            updated_trust = alpha * enhanced_trust + (1 - alpha) * final_trust
        else:
            updated_trust = enhanced_trust
        
        # Ensure trust scores are in valid range
        updated_trust = np.clip(updated_trust, 0.0, 1.0)
        
        logger.info(f"Trust update: mean={np.mean(updated_trust):.4f}, "
                   f"std={np.std(updated_trust):.4f}, iteration={iteration}")
        
        return updated_trust
    
    def _check_convergence(self, trust_scores: np.ndarray, accuracy: float) -> bool:
        """
        Check if trust scores have converged.
        
        Args:
            trust_scores: Current trust scores
            accuracy: Current accuracy
            
        Returns:
            True if converged, False otherwise
        """
        logger = logging.getLogger(__name__)
        
        # Calculate convergence metrics
        mean_trust = np.mean(trust_scores)
        trust_std = np.std(trust_scores)
        
        # Store convergence history
        self._convergence_history.append({
            "mean_trust": mean_trust,
            "trust_std": trust_std,
            "accuracy": accuracy,
            "iteration": len(self._trust_history)
        })
        
        # Enhanced convergence criteria for 10-20 iterations
        min_iterations = 10
        max_iterations = 25  # Reduced from 30 to 25 as requested
        
        current_iteration = len(self._trust_history)
        
        # Don't converge before minimum iterations
        if current_iteration < min_iterations:
            return False
        
        # Force convergence after maximum iterations
        if current_iteration >= max_iterations:
            logger.info(f"Convergence forced after {max_iterations} iterations")
            return True
        
        # Check for trust score stability
        if len(self._trust_history) >= 3:
            recent_trust_means = [h["mean_trust"] for h in self._trust_history[-3:]]
            trust_change = abs(recent_trust_means[-1] - recent_trust_means[0])
            
            # Converge if trust is stable and high
            if trust_change < 0.001 and mean_trust > 0.85:  # Reduced threshold for faster convergence
                logger.info(f"Convergence achieved: trust stable at {mean_trust:.4f}")
                return True
        
        # Check for accuracy improvement plateau
        if len(self._accuracy_history) >= 5:
            recent_accuracies = self._accuracy_history[-5:]
            accuracy_change = abs(recent_accuracies[-1] - recent_accuracies[0])
            
            # Converge if accuracy is stable and high
            if accuracy_change < 0.005 and accuracy > 0.95:
                logger.info(f"Convergence achieved: accuracy stable at {accuracy:.4f}")
                return True
        
        # Check for target trust score achievement
        if mean_trust >= 0.85:  # Reduced from 0.96 to 0.85 as requested
            logger.info(f"Convergence achieved: target trust score {mean_trust:.4f} >= 0.85")
            return True
        
        return False
    
    def get_convergence_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about convergence.
        
        Returns:
            Convergence statistics
        """
        if not self._trust_history:
            return {"message": "No convergence data available"}
        
        # Extract trust values if they are dictionaries
        trust_values = []
        for trust_item in self._trust_history:
            if isinstance(trust_item, dict):
                trust_values.append(trust_item.get('final_trust', 0.0))
            else:
                trust_values.append(trust_item)
        
        accuracy_values = []
        for acc_item in self._accuracy_history:
            if isinstance(acc_item, dict):
                accuracy_values.append(acc_item.get('accuracy', 0.0))
            else:
                accuracy_values.append(acc_item)
        
        stats = {
            "total_iterations": len(self._trust_history),
            "final_trust": trust_values[-1] if trust_values else 0.0,
            "final_accuracy": accuracy_values[-1] if accuracy_values else 0.0,
            "avg_trust": np.mean(trust_values) if trust_values else 0.0,
            "avg_accuracy": np.mean(accuracy_values) if accuracy_values else 0.0,
            "trust_std": np.std(trust_values) if trust_values else 0.0,
            "accuracy_std": np.std(accuracy_values) if accuracy_values else 0.0,
            "convergence_achieved": any(self._convergence_history)
        }
        
        # Add convergence details
        if any(self._convergence_history):
            convergence_iteration = self._convergence_history.index(True) + 1
            stats.update({
                "convergence_iteration": convergence_iteration,
                "iterations_to_convergence": convergence_iteration
            })
        
        return stats
    
    def save_results(self, results: Dict[str, Any], filename: str = "ppp_results.json") -> str:
        """
        Save PPP loop results to disk.
        
        Args:
            results: Results dictionary
            filename: Output filename
            
        Returns:
            Path to saved results
        """
        results_path = LOGS_DIR / filename
        
        # Add metadata
        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "config": self._config,
            "convergence_stats": self.get_convergence_statistics()
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results = convert_numpy_types(results)
        
        # Save results as JSON
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.getLogger(__name__).info(f"Results saved to {results_path}")
        return str(results_path)
    
    def load_results(self, filename: str = "ppp_results.json") -> Dict[str, Any]:
        """
        Load PPP loop results from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            Results dictionary
        """
        results_path = LOGS_DIR / filename
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        logging.getLogger(__name__).info(f"Results loaded from {results_path}")
        return results
    
    def get_validator_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from all validators.
        
        Returns:
            Combined validator metadata
        """
        metadata = {
            "pattern": self._pattern_validator.get_metadata(),
            "presence": self._presence_validator.get_metadata(),
            "permanence": self._permanence_validator.get_metadata(),
            "logic": self._logic_validator.get_metadata(),
            "loop_config": {
                "iterations": self._iterations,
                "gamma": self._gamma,
                "alpha": self._alpha,
                "beta": self._beta,
                "delta": self._delta,
                "initial_trust": self._initial_trust,
                "initial_state": self._initial_state
            }
        }
        
        return metadata
    
    def reset(self):
        """Reset loop state."""
        self._trust_history = []
        self._accuracy_history = []
        self._convergence_history = []
        self._current_trust = self._initial_trust
        self._current_state = self._initial_state
        self._previous_trust = None
        self._previous_state = None
        
        # Reset all validators
        self._pattern_validator.reset()
        self._presence_validator.reset()
        self._permanence_validator.reset()
        self._logic_validator.reset()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current loop state."""
        return {
            "iterations": self._iterations,
            "gamma": self._gamma,
            "alpha": self._alpha,
            "beta": self._beta,
            "delta": self._delta,
            "current_trust": self._current_trust,
            "current_state": self._current_state,
            "trust_history_length": len(self._trust_history),
            "accuracy_history_length": len(self._accuracy_history)
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set loop state."""
        if "iterations" in state:
            self._iterations = state["iterations"]
        if "gamma" in state:
            self._gamma = state["gamma"]
        if "alpha" in state:
            self._alpha = state["alpha"]
        if "beta" in state:
            self._beta = state["beta"]
        if "delta" in state:
            self._delta = state["delta"]
        if "current_trust" in state:
            self._current_trust = state["current_trust"]
        if "current_state" in state:
            self._current_state = state["current_state"]


def create_trust_loop(**kwargs) -> TrustUpdateLoop:
    """
    Factory function to create a Trust Update Loop.
    
    Args:
        **kwargs: Configuration arguments
        
    Returns:
        Configured TrustUpdateLoop instance
    """
    return TrustUpdateLoop(**kwargs) 