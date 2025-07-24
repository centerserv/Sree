"""
SREE Phase 1 Demo - Presence Layer Validator
Entropy minimization for quantum-inspired validation.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from layers.base import Validator
from config import PPP_CONFIG


class PresenceValidator(Validator):
    """
    Presence Layer Validator - Entropy minimization for quantum-inspired validation.
    
    This validator implements the quantum-inspired component of the PPP loop,
    using entropy minimization to refine pattern predictions. It's designed to
    boost accuracy by 4-6% over the Pattern layer baseline.
    
    Target: +4-6% accuracy improvement over Pattern layer
    """
    
    def __init__(self, name: str = "PresenceValidator", **kwargs):
        """
        Initialize Presence validator with entropy minimization.
        
        Args:
            name: Validator name
            **kwargs: Additional configuration parameters
        """
        # Initialize state attributes first
        self._entropy_threshold = PPP_CONFIG["presence"]["entropy_threshold"]
        self._min_confidence = PPP_CONFIG["presence"]["min_confidence"]
        self._entropy_history = []
        self._refinement_count = 0
        
        # Get presence configuration
        presence_config = PPP_CONFIG["presence"].copy()
        presence_config.update(kwargs)
        
        # Store configuration
        self._config = presence_config
        
        # Call parent constructor last
        super().__init__(name=name)
    
    def validate(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Validate data using entropy minimization.
        
        Args:
            data: Input features (n_samples, n_features)
            labels: Optional ground truth labels for evaluation
            
        Returns:
            Trust scores based on entropy minimization
        """
        logger = logging.getLogger(__name__)
        
        # Calculate entropy for each sample
        entropies = self._calculate_entropy(data)
        
        # Apply entropy minimization
        trust_scores = self._minimize_entropy(entropies, data)
        
        # Store entropy history
        self._entropy_history.append({
            "mean_entropy": np.mean(entropies),
            "std_entropy": np.std(entropies),
            "min_entropy": np.min(entropies),
            "max_entropy": np.max(entropies),
            "n_samples": len(data)
        })
        
        logger.info(f"Presence entropy: mean={np.mean(entropies):.4f}, "
                   f"std={np.std(entropies):.4f}")
        
        return trust_scores
    
    def _calculate_entropy(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate entropy for each sample in the data using scipy.stats.entropy.
        
        Args:
            data: Input features (n_samples, n_features)
            
        Returns:
            Entropy values for each sample
        """
        from scipy.stats import entropy
        
        # Normalize data to [0, 1] range for entropy calculation
        data_normalized = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 1e-8)
        
        # Calculate entropy using scipy.stats.entropy
        entropies = np.zeros(len(data))
        
        for i, sample in enumerate(data_normalized):
            # Create histogram for probability distribution
            hist, _ = np.histogram(sample, bins=min(20, len(sample)), density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            
            if len(hist) > 0:
                # Normalize histogram to sum to 1 (probability distribution)
                hist = hist / np.sum(hist)
                # Use scipy.stats.entropy with base=2 for bits
                entropy_val = entropy(hist, base=2)
                # Ensure entropy is non-negative and reasonable
                entropies[i] = max(0.1, min(entropy_val, 10.0))  # Minimum 0.1, cap at 10.0
            else:
                # If no valid histogram, use a small non-zero entropy
                entropies[i] = 0.1
        
        # Ensure we have non-zero entropy values
        if np.all(entropies == 0):
            # Fallback: calculate entropy directly from normalized data
            for i, sample in enumerate(data_normalized):
                # Use variance as a proxy for entropy
                variance = np.var(sample)
                entropies[i] = max(0.1, min(variance * 2, 5.0))
        
        return entropies
    
    def _minimize_entropy(self, entropies: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Apply entropy minimization to generate trust scores.
        
        Args:
            entropies: Entropy values for each sample
            data: Original input features
            
        Returns:
            Trust scores based on entropy minimization
        """
        # Convert entropy to trust scores (lower entropy = higher trust)
        # Use exponential decay: trust = exp(-entropy / threshold)
        trust_scores = np.exp(-entropies / self._entropy_threshold)
        
        # Apply minimum confidence threshold
        trust_scores = np.maximum(trust_scores, self._min_confidence)
        
        # Normalize to [0, 1] range
        trust_scores = np.clip(trust_scores, 0.0, 1.0)
        
        # Count refinements (samples with significant entropy reduction)
        high_entropy_mask = entropies > self._entropy_threshold
        self._refinement_count += np.sum(high_entropy_mask)
        
        return trust_scores
    
    def refine_predictions(self, pattern_predictions: np.ndarray, 
                          pattern_probabilities: np.ndarray,
                          data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine pattern predictions using entropy minimization.
        
        Args:
            pattern_predictions: Predictions from Pattern layer
            pattern_probabilities: Probabilities from Pattern layer
            data: Input features
            
        Returns:
            Refined predictions and probabilities
        """
        logger = logging.getLogger(__name__)
        
        # Calculate entropy-based trust scores
        trust_scores = self.validate(data)
        
        # Apply entropy-based refinement
        refined_probabilities = pattern_probabilities.copy()
        
        # For samples with high entropy (low trust), reduce confidence
        # and potentially adjust predictions
        for i, trust_score in enumerate(trust_scores):
            if trust_score < self._min_confidence:
                # Reduce confidence for uncertain samples
                confidence_factor = trust_score / self._min_confidence
                refined_probabilities[i] *= confidence_factor
                
                # Renormalize probabilities
                refined_probabilities[i] /= np.sum(refined_probabilities[i])
        
        # Get refined predictions
        refined_predictions = np.argmax(refined_probabilities, axis=1)
        
        # Log refinement statistics
        n_refined = np.sum(trust_scores < self._min_confidence)
        logger.info(f"Presence refined {n_refined}/{len(trust_scores)} predictions")
        
        return refined_predictions, refined_probabilities
    
    def get_entropy_statistics(self) -> Dict[str, Any]:
        """
        Get entropy statistics from validation history.
        
        Returns:
            Dictionary with entropy statistics
        """
        if not self._entropy_history:
            return {"message": "No entropy data available"}
        
        recent_entropies = self._entropy_history[-10:]  # Last 10 validations
        
        stats = {
            "mean_entropy": np.mean([e["mean_entropy"] for e in recent_entropies]),
            "std_entropy": np.mean([e["std_entropy"] for e in recent_entropies]),
            "min_entropy": np.min([e["min_entropy"] for e in recent_entropies]),
            "max_entropy": np.max([e["max_entropy"] for e in recent_entropies]),
            "total_refinements": self._refinement_count,
            "n_validations": len(self._entropy_history)
        }
        
        return stats
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get validator metadata including entropy information.
        
        Returns:
            Metadata dictionary
        """
        metadata = super().get_metadata()
        metadata.update({
            "entropy_threshold": self._entropy_threshold,
            "min_confidence": self._min_confidence,
            "total_refinements": self._refinement_count,
            "n_validations": len(self._entropy_history),
            "description": "Entropy minimization for quantum-inspired validation"
        })
        
        # Add entropy statistics if available
        entropy_stats = self.get_entropy_statistics()
        if "message" not in entropy_stats:
            metadata.update({
                "mean_entropy": entropy_stats["mean_entropy"],
                "std_entropy": entropy_stats["std_entropy"]
            })
        
        return metadata
    
    def reset(self):
        """Reset validator state."""
        self._entropy_history = []
        self._refinement_count = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current validator state."""
        state = super().get_state()
        state.update({
            "entropy_threshold": self._entropy_threshold,
            "min_confidence": self._min_confidence,
            "total_refinements": self._refinement_count,
            "n_validations": len(self._entropy_history)
        })
        return state
    
    def set_state(self, state: Dict[str, Any]):
        """Set validator state."""
        super().set_state(state)
        if "entropy_threshold" in state:
            self._entropy_threshold = state["entropy_threshold"]
        if "min_confidence" in state:
            self._min_confidence = state["min_confidence"]
        if "total_refinements" in state:
            self._refinement_count = state["total_refinements"]


def create_presence_validator(**kwargs) -> PresenceValidator:
    """
    Factory function to create a Presence validator.
    
    Args:
        **kwargs: Configuration arguments
        
    Returns:
        Configured PresenceValidator instance
    """
    return PresenceValidator(**kwargs) 