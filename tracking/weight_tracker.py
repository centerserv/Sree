#!/usr/bin/env python3
"""
SREE Weight Change Tracking System
Tracks specific weight changes per feature during training and validation.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class WeightTracker:
    """
    Advanced weight tracking system for SREE features.
    
    Tracks weight changes per feature across iterations, providing insights into:
    - Feature stability and convergence
    - Weight oscillation patterns
    - Feature importance evolution
    - Anomalous weight behavior
    """
    
    def __init__(self, feature_names: List[str], logs_dir: Path = None):
        """
        Initialize Weight Tracker.
        
        Args:
            feature_names: List of feature names to track
            logs_dir: Directory to store weight tracking logs
        """
        self.feature_names = feature_names
        self.logs_dir = logs_dir or Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Weight tracking data
        self.weight_history = {name: [] for name in feature_names}
        self.iteration_history = []
        self.change_history = {name: [] for name in feature_names}
        self.stability_scores = {name: 0.0 for name in feature_names}
        
        # Statistics
        self.feature_stats = {}
        self.anomaly_detection = {}
        
        self.logger = logging.getLogger(__name__)
        
    def track_weights(self, iteration: int, weights: np.ndarray, 
                     trust_scores: np.ndarray = None, 
                     accuracy: float = None) -> Dict[str, Any]:
        """
        Track weight changes for current iteration.
        
        Args:
            iteration: Current iteration number
            weights: Feature weights array
            trust_scores: Trust scores for each sample
            accuracy: Current accuracy
            
        Returns:
            Dictionary with tracking information
        """
        if len(weights) != len(self.feature_names):
            raise ValueError(f"Weights length ({len(weights)}) doesn't match feature names ({len(self.feature_names)})")
        
        # Store iteration info
        iteration_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "avg_trust": float(np.mean(trust_scores)) if trust_scores is not None else None,
            "weights": {}
        }
        
        # Track weights for each feature
        for i, feature_name in enumerate(self.feature_names):
            weight = float(weights[i])
            self.weight_history[feature_name].append(weight)
            
            iteration_data["weights"][feature_name] = weight
            
            # Calculate weight change
            if len(self.weight_history[feature_name]) > 1:
                change = weight - self.weight_history[feature_name][-2]
                self.change_history[feature_name].append(change)
            else:
                self.change_history[feature_name].append(0.0)
        
        self.iteration_history.append(iteration_data)
        
        # Update stability scores
        self._update_stability_scores()
        
        # Detect anomalies
        anomalies = self._detect_weight_anomalies()
        
        return {
            "iteration_data": iteration_data,
            "stability_scores": self.stability_scores.copy(),
            "anomalies": anomalies
        }
    
    def _update_stability_scores(self):
        """Update stability scores for each feature."""
        for feature_name in self.feature_names:
            if len(self.weight_history[feature_name]) > 1:
                weights = np.array(self.weight_history[feature_name])
                changes = np.array(self.change_history[feature_name][1:])  # Skip first change
                
                # Calculate stability metrics
                weight_std = np.std(weights)
                change_std = np.std(changes)
                weight_range = np.max(weights) - np.min(weights)
                
                # Stability score: lower is more stable
                stability = (weight_std + change_std + weight_range) / 3.0
                self.stability_scores[feature_name] = max(0.0, 1.0 - stability)
    
    def _detect_weight_anomalies(self) -> Dict[str, List[Dict]]:
        """Detect anomalous weight behavior."""
        anomalies = {}
        
        for feature_name in self.feature_names:
            feature_anomalies = []
            weights = np.array(self.weight_history[feature_name])
            
            if len(weights) < 3:
                continue
            
            # Detect sudden large changes
            changes = np.diff(weights)
            mean_change = np.mean(np.abs(changes))
            std_change = np.std(changes)
            
            for i, change in enumerate(changes):
                if abs(change) > mean_change + 2 * std_change:
                    feature_anomalies.append({
                        "type": "sudden_change",
                        "iteration": i + 1,
                        "change": float(change),
                        "threshold": float(mean_change + 2 * std_change),
                        "description": f"Sudden weight change of {change:.4f}"
                    })
            
            # Detect oscillation patterns
            if len(changes) > 5:
                oscillation_score = np.sum(np.abs(np.diff(changes))) / len(changes)
                if oscillation_score > 0.1:  # High oscillation threshold
                    feature_anomalies.append({
                        "type": "oscillation",
                        "score": float(oscillation_score),
                        "description": f"High oscillation detected (score: {oscillation_score:.4f})"
                    })
            
            # Detect convergence issues
            recent_weights = weights[-5:] if len(weights) >= 5 else weights
            if len(recent_weights) >= 3:
                recent_std = np.std(recent_weights)
                if recent_std > 0.05:  # Not converging
                    feature_anomalies.append({
                        "type": "non_convergence",
                        "recent_std": float(recent_std),
                        "description": f"Weight not converging (std: {recent_std:.4f})"
                    })
            
            if feature_anomalies:
                anomalies[feature_name] = feature_anomalies
        
        return anomalies
    
    def get_feature_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about feature weights."""
        insights = {}
        
        for feature_name in self.feature_names:
            weights = np.array(self.weight_history[feature_name])
            
            if len(weights) == 0:
                continue
            
            # Basic statistics
            stats = {
                "current_weight": float(weights[-1]),
                "initial_weight": float(weights[0]),
                "min_weight": float(np.min(weights)),
                "max_weight": float(np.max(weights)),
                "mean_weight": float(np.mean(weights)),
                "std_weight": float(np.std(weights)),
                "weight_range": float(np.max(weights) - np.min(weights)),
                "stability_score": self.stability_scores[feature_name],
                "total_change": float(weights[-1] - weights[0]),
                "change_percentage": float((weights[-1] - weights[0]) / weights[0] * 100) if weights[0] != 0 else 0.0
            }
            
            # Trend analysis
            if len(weights) > 1:
                changes = np.diff(weights)
                stats.update({
                    "avg_change": float(np.mean(changes)),
                    "change_std": float(np.std(changes)),
                    "positive_changes": int(np.sum(changes > 0)),
                    "negative_changes": int(np.sum(changes < 0)),
                    "zero_changes": int(np.sum(changes == 0))
                })
                
                # Trend direction
                if len(changes) >= 3:
                    recent_trend = np.mean(changes[-3:])
                    if recent_trend > 0.01:
                        stats["trend"] = "increasing"
                    elif recent_trend < -0.01:
                        stats["trend"] = "decreasing"
                    else:
                        stats["trend"] = "stable"
                else:
                    stats["trend"] = "insufficient_data"
            
            insights[feature_name] = stats
        
        return insights
    
    def generate_weight_visualization(self, save_path: str = None) -> str:
        """Generate weight evolution visualization."""
        if not self.weight_history or not any(self.weight_history.values()):
            return ""
        
        # Create subplots
        n_features = len(self.feature_names)
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Weight evolution plot
        ax1 = axes[0]
        for feature_name in self.feature_names:
            weights = self.weight_history[feature_name]
            if weights:
                ax1.plot(range(len(weights)), weights, label=feature_name, marker='o', markersize=3)
        
        ax1.set_title('Feature Weight Evolution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Weight Value')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Stability scores plot
        ax2 = axes[1]
        features = list(self.stability_scores.keys())
        scores = list(self.stability_scores.values())
        
        colors = ['green' if score > 0.7 else 'orange' if score > 0.4 else 'red' for score in scores]
        bars = ax2.bar(features, scores, color=colors, alpha=0.7)
        
        ax2.set_title('Feature Stability Scores', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Stability Score (0-1)')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.logs_dir / f"weight_tracking_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def save_weight_logs(self, filename: str = None) -> str:
        """Save weight tracking logs to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weight_tracking_logs_{timestamp}.json"
        
        filepath = self.logs_dir / filename
        
        logs_data = {
            "feature_names": self.feature_names,
            "weight_history": self.weight_history,
            "iteration_history": self.iteration_history,
            "change_history": self.change_history,
            "stability_scores": self.stability_scores,
            "feature_insights": self.get_feature_insights(),
            "anomaly_detection": self._detect_weight_anomalies(),
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_iterations": len(self.iteration_history),
                "tracking_duration": self._calculate_tracking_duration()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(logs_data, f, indent=2, default=str)
        
        return str(filepath)
    
    def _calculate_tracking_duration(self) -> str:
        """Calculate tracking duration."""
        if len(self.iteration_history) < 2:
            return "0 seconds"
        
        start_time = datetime.fromisoformat(self.iteration_history[0]["timestamp"])
        end_time = datetime.fromisoformat(self.iteration_history[-1]["timestamp"])
        duration = end_time - start_time
        
        return str(duration)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get a summary report of weight tracking."""
        insights = self.get_feature_insights()
        
        # Calculate overall statistics
        stability_scores = list(self.stability_scores.values())
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        # Identify most and least stable features
        if stability_scores:
            most_stable = min(self.stability_scores.items(), key=lambda x: x[1])
            least_stable = max(self.stability_scores.items(), key=lambda x: x[1])
        else:
            most_stable = least_stable = ("N/A", 0.0)
        
        # Count anomalies
        anomalies = self._detect_weight_anomalies()
        total_anomalies = sum(len(anomaly_list) for anomaly_list in anomalies.values())
        
        return {
            "total_features": len(self.feature_names),
            "total_iterations": len(self.iteration_history),
            "average_stability": float(avg_stability),
            "most_stable_feature": {
                "name": most_stable[0],
                "stability_score": most_stable[1]
            },
            "least_stable_feature": {
                "name": least_stable[0],
                "stability_score": least_stable[1]
            },
            "total_anomalies": total_anomalies,
            "features_with_anomalies": len(anomalies),
            "tracking_duration": self._calculate_tracking_duration(),
            "feature_insights": insights
        } 