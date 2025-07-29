#!/usr/bin/env python3
"""
Adaptive Threshold Evaluation System for SREE
Implements soft evaluation layer with weighted scoring and explainable classification.
"""

import logging
from typing import List, Optional, Literal, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class MetricStatus(Enum):
    """Status levels for metric evaluation."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class OverallStatus(Enum):
    """Overall evaluation status."""
    EXCELLENT = "excellent"
    ACCEPTABLE = "acceptable"
    FAIL = "fail"


@dataclass
class MetricEvaluation:
    """Individual metric evaluation result."""
    metric: str
    value: float
    threshold: float
    status: MetricStatus
    reason: Optional[str] = None
    weight: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "status": self.status.value,
            "reason": self.reason,
            "weight": self.weight
        }


@dataclass
class AdaptiveEvaluationResult:
    """Complete adaptive evaluation result."""
    final_score: float
    status: OverallStatus
    breakdown: List[MetricEvaluation]
    industry: str
    timestamp: str
    auto_refinement_triggered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "final_score": self.final_score,
            "status": self.status.value,
            "industry": self.industry,
            "timestamp": self.timestamp,
            "auto_refinement_triggered": self.auto_refinement_triggered,
            "breakdown": [metric.to_dict() for metric in self.breakdown]
        }


class AdaptiveThresholdEvaluator:
    """
    Adaptive threshold evaluator with soft zones and weighted scoring.
    """
    
    def __init__(self, industry_config: Dict[str, Any]):
        """
        Initialize evaluator with industry-specific configuration.
        
        Args:
            industry_config: Industry configuration with thresholds and weights
        """
        self.industry_config = industry_config
        self.logger = logging.getLogger(__name__)
        
        # Default weights (can be overridden by industry config)
        self.weights = {
            "accuracy": 0.4,
            "trust": 0.3,
            "entropy": 0.2,
            "block_count": 0.1
        }
        
        # Update weights if specified in industry config
        if "weights" in industry_config:
            self.weights.update(industry_config["weights"])
    
    def classify_metric(self, metric: str, value: float, threshold: float, 
                       max_value: Optional[float] = None) -> MetricEvaluation:
        """
        Classify a single metric with soft zones.
        
        Args:
            metric: Metric name (accuracy, trust, entropy, block_count)
            value: Actual metric value
            threshold: Target threshold
            max_value: Maximum possible value (for normalization)
            
        Returns:
            MetricEvaluation with status and reason
        """
        # Determine classification logic based on metric type
        if metric == "entropy":
            # For entropy: lower is better, so we invert the logic
            # Round to 3 decimal places for comparison (matching display precision)
            value_rounded = round(value, 3)
            threshold_rounded = round(threshold, 3)
            
            if value_rounded <= threshold_rounded:
                status = MetricStatus.PASS
                reason = f"Entropy {value_rounded:.3f} is within acceptable range (≤{threshold_rounded:.3f})"
            elif value_rounded <= threshold_rounded * 1.25:  # 25% tolerance for entropy
                status = MetricStatus.WARN
                reason = f"Entropy {value_rounded:.3f} is slightly high (target ≤{threshold_rounded:.3f})"
            else:
                status = MetricStatus.FAIL
                reason = f"Entropy {value_rounded:.3f} is too high (target ≤{threshold_rounded:.3f})"
                
        elif metric == "block_count":
            # For block count: lower is better, with max limit
            max_blocks = self.industry_config.get("max_blocks", 25)
            if value <= max_blocks * 0.8:  # 80% of max is excellent
                status = MetricStatus.PASS
                reason = f"Block count {value} is efficient (≤{max_blocks * 0.8:.0f})"
            elif value <= max_blocks:
                status = MetricStatus.WARN
                reason = f"Block count {value} is near limit ({max_blocks})"
            else:
                status = MetricStatus.FAIL
                reason = f"Block count {value} exceeds limit ({max_blocks})"
                
        else:
            # For accuracy and trust: higher is better
            # Round to 3 decimal places for comparison (matching display precision)
            value_rounded = round(value, 3)
            threshold_rounded = round(threshold, 3)
            
            if value_rounded >= threshold_rounded:
                status = MetricStatus.PASS
                reason = f"{metric.title()} {value_rounded:.3f} meets target (≥{threshold_rounded:.3f})"
            elif value_rounded >= threshold_rounded * 0.95:  # 95% of threshold is warning zone
                status = MetricStatus.WARN
                reason = f"{metric.title()} {value_rounded:.3f} is close to target (≥{threshold_rounded:.3f})"
            else:
                status = MetricStatus.FAIL
                reason = f"{metric.title()} {value_rounded:.3f} is below target (≥{threshold_rounded:.3f})"
        
        return MetricEvaluation(
            metric=metric,
            value=value,
            threshold=threshold,
            status=status,
            reason=reason,
            weight=self.weights.get(metric, 0.0)
        )
    
    def compute_final_score(self, metrics: List[MetricEvaluation]) -> float:
        """
        Compute weighted final score from individual metrics.
        
        Args:
            metrics: List of metric evaluations
            
        Returns:
            Final score between 0.0 and 1.0
        """
        total_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            weight = metric.weight
            
            if metric.metric == "entropy":
                # For entropy: normalize to 0-1 range (lower is better)
                max_entropy = self.industry_config.get("entropy_threshold", 1.5) * 2
                normalized_value = max(0, 1 - (metric.value / max_entropy))
                score = normalized_value
                
            elif metric.metric == "block_count":
                # For block count: normalize to 0-1 range (lower is better)
                max_blocks = self.industry_config.get("max_blocks", 25)
                normalized_value = max(0, 1 - (metric.value / max_blocks))
                score = normalized_value
                
            else:
                # For accuracy and trust: use value directly
                score = min(1.0, metric.value)
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def determine_overall_status(self, final_score: float) -> OverallStatus:
        """
        Determine overall status based on final score.
        
        Args:
            final_score: Computed final score
            
        Returns:
            Overall status
        """
        if final_score >= 0.85:
            return OverallStatus.EXCELLENT
        elif final_score >= 0.70:
            return OverallStatus.ACCEPTABLE
        else:
            return OverallStatus.FAIL
    
    def evaluate(self, results: Dict[str, Any], industry: str) -> AdaptiveEvaluationResult:
        """
        Perform complete adaptive evaluation.
        
        Args:
            results: SREE analysis results
            industry: Industry name
            
        Returns:
            Complete evaluation result
        """
        self.logger.info(f"🔍 Starting adaptive evaluation for {industry}")
        
        # Extract metrics from results
        accuracy = results.get('accuracy', 0.0)
        trust_score = results.get('trust_score', 0.0)
        entropy = results.get('entropy', 0.0)
        block_count = results.get('block_count', 0)
        
        # Get thresholds from industry config
        accuracy_threshold = self.industry_config.get('accuracy_threshold', 0.95)
        trust_threshold = self.industry_config.get('trust_threshold', 0.85)
        entropy_threshold = self.industry_config.get('entropy_threshold', 1.5)
        
        # Classify each metric
        metrics = [
            self.classify_metric("accuracy", accuracy, accuracy_threshold),
            self.classify_metric("trust", trust_score, trust_threshold),
            self.classify_metric("entropy", entropy, entropy_threshold),
            self.classify_metric("block_count", block_count, 0)  # threshold handled internally
        ]
        
        # Compute final score
        final_score = self.compute_final_score(metrics)
        
        # Determine overall status
        overall_status = self.determine_overall_status(final_score)
        
        # Check if auto-refinement should be triggered
        auto_refinement_triggered = (
            overall_status == OverallStatus.FAIL and 
            final_score > 0.70 and
            self.industry_config.get("auto_refinement", False)
        )
        
        # Create evaluation result
        evaluation_result = AdaptiveEvaluationResult(
            final_score=final_score,
            status=overall_status,
            breakdown=metrics,
            industry=industry,
            timestamp=datetime.now().isoformat(),
            auto_refinement_triggered=auto_refinement_triggered
        )
        
        # Log evaluation summary
        self._log_evaluation_summary(evaluation_result)
        
        return evaluation_result
    
    def _log_evaluation_summary(self, result: AdaptiveEvaluationResult):
        """Log detailed evaluation summary."""
        self.logger.info(f"📊 ADAPTIVE EVALUATION SUMMARY:")
        self.logger.info(f"   Industry: {result.industry}")
        self.logger.info(f"   Final Score: {result.final_score:.3f}")
        self.logger.info(f"   Overall Status: {result.status.value.upper()}")
        self.logger.info(f"   Auto-refinement: {'🔧 TRIGGERED' if result.auto_refinement_triggered else '✅ NOT NEEDED'}")
        
        self.logger.info(f"   Metric Breakdown:")
        for metric in result.breakdown:
            status_icon = {
                MetricStatus.PASS: "✅",
                MetricStatus.WARN: "⚠️",
                MetricStatus.FAIL: "❌"
            }[metric.status]
            
            self.logger.info(f"     {status_icon} {metric.metric}: {metric.value:.3f} → {metric.reason}")
    
    def save_evaluation(self, result: AdaptiveEvaluationResult, output_file: str = None):
        """
        Save evaluation result to JSON file.
        
        Args:
            result: Evaluation result to save
            output_file: Output file path (optional)
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_{result.industry}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self.logger.info(f"💾 Evaluation saved to: {output_file}")
        return output_file


def create_adaptive_evaluator(industry_config: Dict[str, Any]) -> AdaptiveThresholdEvaluator:
    """
    Factory function to create adaptive evaluator with industry configuration.
    
    Args:
        industry_config: Industry-specific configuration
        
    Returns:
        Configured adaptive evaluator
    """
    return AdaptiveThresholdEvaluator(industry_config) 