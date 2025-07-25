"""
SREE Phase 1 Demo - Logic Layer Validator
Consistency validation for logical coherence across PPP components.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from layers.base import Validator
from config import PPP_CONFIG


class LogicValidator(Validator):
    """
    Logic Layer Validator - Consistency validation for logical coherence.
    
    This validator implements the logical consistency component of the PPP loop,
    ensuring that predictions and validations from all layers are coherent and
    logically consistent. It acts as a final validation layer that checks for
    contradictions and ensures overall system reliability.
    
    Target: Ensure logical consistency and coherence across all PPP components
    """
    
    def __init__(self, name: str = "LogicValidator", **kwargs):
        """
        Initialize Logic validator with consistency validation.
        
        Args:
            name: Validator name
            **kwargs: Additional configuration parameters
        """
        # Initialize state attributes first
        self._consistency_weight = PPP_CONFIG["logic"]["consistency_weight"]
        self._confidence_threshold = PPP_CONFIG["logic"]["confidence_threshold"]
        self._max_inconsistencies = PPP_CONFIG["logic"]["max_inconsistencies"]
        self._validation_history = []
        self._inconsistency_counts = []
        self._consistency_scores = []
        
        # Get logic configuration
        logic_config = PPP_CONFIG["logic"].copy()
        logic_config.update(kwargs)
        
        # Store configuration
        self._config = logic_config
        
        # Call parent constructor last
        super().__init__(name=name)
    
    def validate(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Validate data using logical consistency checks.
        
        Args:
            data: Input features (n_samples, n_features)
            labels: Optional ground truth labels for evaluation
            
        Returns:
            Trust scores based on logical consistency
        """
        logger = logging.getLogger(__name__)
        
        # Calculate basic consistency metrics
        consistency_scores = self._calculate_consistency_scores(data, labels)
        
        # Apply confidence threshold filtering
        filtered_scores = self._apply_confidence_filter(consistency_scores, data)
        
        # Calculate final trust scores
        trust_scores = self._calculate_trust_scores(filtered_scores, data)
        
        # Store validation history
        self._validation_history.append({
            "n_samples": len(data),
            "avg_consistency": float(np.mean(consistency_scores)),
            "avg_trust": float(np.mean(trust_scores)),
            "n_inconsistencies": len([s for s in consistency_scores if s < self._confidence_threshold])
        })
        
        logger.info(f"Logic validated {len(data)} samples, "
                   f"avg consistency: {np.mean(consistency_scores):.4f}, "
                   f"avg trust: {np.mean(trust_scores):.4f}")
        
        return trust_scores
    
    def _calculate_consistency_scores(self, data: np.ndarray, 
                                    labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate consistency scores for the data.
        
        Args:
            data: Input features
            labels: Optional ground truth labels
            
        Returns:
            Consistency scores for each sample
        """
        n_samples = len(data)
        consistency_scores = np.ones(n_samples)
        
        # Check feature consistency
        feature_consistency = self._check_feature_consistency(data)
        consistency_scores *= feature_consistency
        
        # Check label consistency if available
        if labels is not None:
            label_consistency = self._check_label_consistency(data, labels)
            consistency_scores *= label_consistency
        
        # Check data distribution consistency
        distribution_consistency = self._check_distribution_consistency(data)
        consistency_scores *= distribution_consistency
        
        # Store consistency scores
        self._consistency_scores.extend(consistency_scores.tolist())
        
        return consistency_scores
    
    def _check_feature_consistency(self, data: np.ndarray) -> np.ndarray:
        """
        Check consistency of individual features.
        
        Args:
            data: Input features
            
        Returns:
            Feature consistency scores
        """
        n_samples = len(data)
        consistency_scores = np.ones(n_samples)
        
        # Check for NaN or infinite values
        nan_mask = np.isnan(data).any(axis=1)
        inf_mask = np.isinf(data).any(axis=1)
        
        # Reduce consistency for samples with invalid values
        consistency_scores[nan_mask] *= 0.5
        consistency_scores[inf_mask] *= 0.5
        
        # Check for extreme outliers
        for i in range(n_samples):
            sample = data[i]
            # Calculate z-scores
            z_scores = np.abs((sample - np.mean(sample)) / (np.std(sample) + 1e-8))
            # Count extreme outliers (z-score > 3)
            extreme_outliers = np.sum(z_scores > 3)
            if extreme_outliers > len(sample) * 0.1:  # More than 10% extreme outliers
                consistency_scores[i] *= 0.8
        
        return consistency_scores
    
    def _check_label_consistency(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Check consistency between features and labels.
        
        Args:
            data: Input features
            labels: Ground truth labels
            
        Returns:
            Label consistency scores
        """
        n_samples = len(data)
        consistency_scores = np.ones(n_samples)
        
        # Check for label distribution consistency
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        
        # If labels are too imbalanced, reduce consistency
        if len(unique_labels) > 1:
            min_count = np.min(label_counts)
            max_count = np.max(label_counts)
            imbalance_ratio = min_count / max_count
            
            if imbalance_ratio < 0.1:  # Very imbalanced
                consistency_scores *= 0.9
        
        # Check for label-feature correlation
        for i in range(n_samples):
            sample = data[i]
            label = labels[i]
            
            # Calculate correlation between sample and label
            # For simplicity, use the mean of the sample as a proxy
            sample_mean = np.mean(sample)
            
            # Check if sample mean is reasonable for the label
            # This is a simplified check - in practice, you'd use more sophisticated methods
            if label == 0 and sample_mean > 0.8:
                consistency_scores[i] *= 0.9
            elif label == 1 and sample_mean < 0.2:
                consistency_scores[i] *= 0.9
        
        return consistency_scores
    
    def _check_distribution_consistency(self, data: np.ndarray) -> np.ndarray:
        """
        Check consistency of data distribution.
        
        Args:
            data: Input features
            
        Returns:
            Distribution consistency scores
        """
        n_samples = len(data)
        consistency_scores = np.ones(n_samples)
        
        # Calculate overall statistics
        overall_mean = np.mean(data)
        overall_std = np.std(data)
        
        # Check each sample's consistency with overall distribution
        for i in range(n_samples):
            sample = data[i]
            sample_mean = np.mean(sample)
            sample_std = np.std(sample)
            
            # Check if sample statistics are reasonable compared to overall
            mean_diff = abs(sample_mean - overall_mean) / (overall_std + 1e-8)
            std_diff = abs(sample_std - overall_std) / (overall_std + 1e-8)
            
            # Reduce consistency if sample is too different
            if mean_diff > 2.0:  # More than 2 standard deviations
                consistency_scores[i] *= 0.8
            if std_diff > 1.0:  # Standard deviation too different
                consistency_scores[i] *= 0.9
        
        return consistency_scores
    
    def _apply_confidence_filter(self, consistency_scores: np.ndarray, 
                                data: np.ndarray) -> np.ndarray:
        """
        Apply confidence threshold filtering.
        
        Args:
            consistency_scores: Raw consistency scores
            data: Input features
            
        Returns:
            Filtered consistency scores
        """
        filtered_scores = consistency_scores.copy()
        
        # Count samples below confidence threshold
        low_confidence_mask = consistency_scores < self._confidence_threshold
        n_low_confidence = np.sum(low_confidence_mask)
        
        # Store inconsistency count
        self._inconsistency_counts.append(n_low_confidence)
        
        # If too many inconsistencies, reduce all scores
        if n_low_confidence > len(data) * self._max_inconsistencies:
            filtered_scores *= 0.8
        
        return filtered_scores
    
    def _calculate_trust_scores(self, consistency_scores: np.ndarray, 
                               data: np.ndarray) -> np.ndarray:
        """
        Calculate final trust scores from consistency scores.
        
        Args:
            consistency_scores: Filtered consistency scores
            data: Input features
            
        Returns:
            Final trust scores
        """
        # Apply consistency weight
        trust_scores = consistency_scores * self._consistency_weight
        
        # Add base trust for all samples
        base_trust = (1.0 - self._consistency_weight) * np.ones(len(data))
        trust_scores += base_trust
        
        # Ensure scores are in [0, 1] range
        trust_scores = np.clip(trust_scores, 0.0, 1.0)
        
        return trust_scores
    
    def validate_predictions(self, predictions: np.ndarray, 
                           probabilities: np.ndarray,
                           pattern_trust: np.ndarray,
                           presence_trust: np.ndarray,
                           permanence_trust: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate predictions from all PPP layers for logical consistency.
        
        Args:
            predictions: Pattern layer predictions
            probabilities: Pattern layer probabilities
            pattern_trust: Pattern layer trust scores
            presence_trust: Presence layer trust scores
            permanence_trust: Permanence layer trust scores
            
        Returns:
            Tuple of (validated_predictions, final_trust_scores)
        """
        n_samples = len(predictions)
        validated_predictions = predictions.copy()
        final_trust_scores = np.ones(n_samples)
        
        # Check for prediction consistency across layers
        for i in range(n_samples):
            # Get trust scores for this sample
            pattern_t = pattern_trust[i]
            presence_t = presence_trust[i]
            permanence_t = permanence_trust[i]
            
            # Calculate weighted trust score
            weighted_trust = (pattern_t + presence_t + permanence_t) / 3
            
            # Check probability consistency
            prob = probabilities[i]
            max_prob = np.max(prob)
            
            # If confidence is low, reduce trust
            if max_prob < self._confidence_threshold:
                weighted_trust *= 0.8
            
            # Check for prediction conflicts
            # This is a simplified check - in practice, you'd compare with other layers
            if max_prob < 0.5:  # Low confidence prediction
                weighted_trust *= 0.9
            
            final_trust_scores[i] = weighted_trust
        
        return validated_predictions, final_trust_scores
    
    def get_consistency_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about consistency validation.
        
        Returns:
            Consistency statistics
        """
        if not self._consistency_scores:
            return {"message": "No consistency data available"}
        
        stats = {
            "total_validations": len(self._validation_history),
            "avg_consistency_score": np.mean(self._consistency_scores),
            "min_consistency_score": np.min(self._consistency_scores),
            "max_consistency_score": np.max(self._consistency_scores),
            "consistency_std": np.std(self._consistency_scores)
        }
        
        # Add validation history statistics
        if self._validation_history:
            recent_validations = self._validation_history[-10:]  # Last 10 validations
            stats.update({
                "avg_samples_per_validation": np.mean([v["n_samples"] for v in recent_validations]),
                "avg_consistency_per_validation": np.mean([v["avg_consistency"] for v in recent_validations]),
                "avg_trust_per_validation": np.mean([v["avg_trust"] for v in recent_validations]),
                "avg_inconsistencies_per_validation": np.mean([v["n_inconsistencies"] for v in recent_validations])
            })
        
        # Add inconsistency statistics
        if self._inconsistency_counts:
            stats.update({
                "total_inconsistencies": sum(self._inconsistency_counts),
                "avg_inconsistencies_per_validation": np.mean(self._inconsistency_counts),
                "max_inconsistencies_per_validation": np.max(self._inconsistency_counts)
            })
        
        return stats
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get validator metadata including consistency information.
        
        Returns:
            Metadata dictionary
        """
        metadata = super().get_metadata()
        metadata.update({
            "consistency_weight": self._consistency_weight,
            "confidence_threshold": self._confidence_threshold,
            "max_inconsistencies": self._max_inconsistencies,
            "n_validations": len(self._validation_history),
            "n_consistency_scores": len(self._consistency_scores),
            "description": "Logical consistency validation for PPP coherence"
        })
        
        # Add consistency statistics if available
        consistency_stats = self.get_consistency_statistics()
        if "message" not in consistency_stats:
            metadata.update({
                "avg_consistency_score": consistency_stats["avg_consistency_score"],
                "avg_inconsistencies_per_validation": consistency_stats["avg_inconsistencies_per_validation"]
            })
        
        return metadata
    
    def reset(self):
        """Reset validator state."""
        self._validation_history = []
        self._inconsistency_counts = []
        self._consistency_scores = []
    
    def get_state(self) -> Dict[str, Any]:
        """Get current validator state."""
        state = super().get_state()
        state.update({
            "consistency_weight": self._consistency_weight,
            "confidence_threshold": self._confidence_threshold,
            "max_inconsistencies": self._max_inconsistencies,
            "n_validations": len(self._validation_history),
            "n_consistency_scores": len(self._consistency_scores)
        })
        return state
    
    def set_state(self, state: Dict[str, Any]):
        """Set validator state."""
        super().set_state(state)
        if "consistency_weight" in state:
            self._consistency_weight = state["consistency_weight"]
        if "confidence_threshold" in state:
            self._confidence_threshold = state["confidence_threshold"]
        if "max_inconsistencies" in state:
            self._max_inconsistencies = state["max_inconsistencies"]


def create_logic_validator(**kwargs) -> LogicValidator:
    """
    Factory function to create a Logic validator.
    
    Args:
        **kwargs: Configuration arguments
        
    Returns:
        Configured LogicValidator instance
    """
    return LogicValidator(**kwargs) 