#!/usr/bin/env python3
"""
SREE Column Revaluation History System
Tracks revaluation history for each column during analysis.
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
from enum import Enum

class RevaluationReason(Enum):
    """Enumeration of revaluation reasons."""
    OUTLIER_DETECTED = "outlier_detected"
    INCONSISTENCY_FOUND = "inconsistency_found"
    MISSING_VALUES = "missing_values"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    RANGE_VIOLATION = "range_violation"
    PATTERN_ANOMALY = "pattern_anomaly"
    TRUST_SCORE_LOW = "trust_score_low"
    LOGIC_RULE_FAILURE = "logic_rule_failure"
    ENTROPY_HIGH = "entropy_high"
    CORRELATION_ANOMALY = "correlation_anomaly"

class ColumnHistory:
    """
    Advanced column revaluation history tracking system.
    
    Tracks revaluation history for each column, providing insights into:
    - Column reliability and trustworthiness
    - Revaluation frequency and patterns
    - Data quality issues over time
    - Column confidence scores
    """
    
    def __init__(self, column_names: List[str], logs_dir: Path = None):
        """
        Initialize Column History Tracker.
        
        Args:
            column_names: List of column names to track
            logs_dir: Directory to store column history logs
        """
        self.column_names = column_names
        self.logs_dir = logs_dir or Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Column tracking data
        self.revaluation_history = {name: [] for name in column_names}
        self.column_stats = {name: {} for name in column_names}
        self.confidence_scores = {name: 1.0 for name in column_names}
        self.issue_patterns = {name: {} for name in column_names}
        
        # Global statistics
        self.total_revaluations = 0
        self.session_start = datetime.now()
        
        self.logger = logging.getLogger(__name__)
        
    def log_revaluation(self, column_name: str, reason: RevaluationReason, 
                       details: Dict[str, Any], iteration: int = None,
                       affected_rows: List[int] = None, 
                       trust_impact: float = None) -> Dict[str, Any]:
        """
        Log a column revaluation event.
        
        Args:
            column_name: Name of the column being revaluated
            reason: Reason for revaluation
            details: Additional details about the revaluation
            iteration: Current iteration number
            affected_rows: List of row indices affected
            trust_impact: Impact on trust score
            
        Returns:
            Dictionary with revaluation information
        """
        if column_name not in self.column_names:
            self.logger.warning(f"Column {column_name} not in tracking list, adding it")
            self.column_names.append(column_name)
            self.revaluation_history[column_name] = []
            self.column_stats[column_name] = {}
            self.confidence_scores[column_name] = 1.0
            self.issue_patterns[column_name] = {}
        
        # Create revaluation record
        revaluation_record = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason.value,
            "details": details,
            "iteration": iteration,
            "affected_rows": affected_rows or [],
            "trust_impact": trust_impact,
            "session_duration": str(datetime.now() - self.session_start)
        }
        
        # Add to history
        self.revaluation_history[column_name].append(revaluation_record)
        
        # Update statistics
        self._update_column_stats(column_name, revaluation_record)
        
        # Update confidence score
        self._update_confidence_score(column_name)
        
        # Update global stats
        self.total_revaluations += 1
        
        return revaluation_record
    
    def _update_column_stats(self, column_name: str, revaluation_record: Dict[str, Any]):
        """Update column statistics."""
        history = self.revaluation_history[column_name]
        
        # Basic counts
        stats = {
            "total_revaluations": len(history),
            "revaluation_frequency": len(history) / max(1, (datetime.now() - self.session_start).total_seconds() / 60),  # per minute
            "last_revaluation": revaluation_record["timestamp"],
            "reasons": {},
            "avg_trust_impact": 0.0,
            "total_affected_rows": 0
        }
        
        # Count reasons
        for record in history:
            reason = record["reason"]
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            
            if record["trust_impact"] is not None:
                stats["avg_trust_impact"] += record["trust_impact"]
            
            stats["total_affected_rows"] += len(record.get("affected_rows", []))
        
        # Calculate averages
        if history:
            stats["avg_trust_impact"] /= len(history)
        
        self.column_stats[column_name] = stats
    
    def _update_confidence_score(self, column_name: str):
        """Update column confidence score based on revaluation history."""
        history = self.revaluation_history[column_name]
        
        if not history:
            self.confidence_scores[column_name] = 1.0
            return
        
        # Base confidence starts at 1.0
        confidence = 1.0
        
        # Penalize based on revaluation frequency
        revaluation_count = len(history)
        frequency_penalty = min(0.5, revaluation_count * 0.1)  # Max 50% penalty
        
        # Penalize based on severity of issues
        severity_penalty = 0.0
        for record in history:
            reason = record["reason"]
            if reason in ["outlier_detected", "inconsistency_found", "logic_rule_failure"]:
                severity_penalty += 0.05
            elif reason in ["missing_values", "data_type_mismatch"]:
                severity_penalty += 0.03
            else:
                severity_penalty += 0.01
        
        severity_penalty = min(0.3, severity_penalty)  # Max 30% penalty
        
        # Penalize based on trust impact
        trust_penalty = 0.0
        trust_impacts = [r["trust_impact"] for r in history if r["trust_impact"] is not None]
        if trust_impacts:
            avg_trust_impact = np.mean(trust_impacts)
            trust_penalty = min(0.2, abs(avg_trust_impact) * 0.5)  # Max 20% penalty
        
        # Calculate final confidence
        confidence = max(0.1, 1.0 - frequency_penalty - severity_penalty - trust_penalty)
        self.confidence_scores[column_name] = confidence
    
    def get_column_insights(self, column_name: str) -> Dict[str, Any]:
        """Get comprehensive insights about a specific column."""
        if column_name not in self.column_names:
            return {}
        
        history = self.revaluation_history[column_name]
        stats = self.column_stats[column_name]
        
        insights = {
            "column_name": column_name,
            "confidence_score": self.confidence_scores[column_name],
            "total_revaluations": len(history),
            "revaluation_frequency": stats.get("revaluation_frequency", 0),
            "most_common_reason": self._get_most_common_reason(column_name),
            "trust_impact": stats.get("avg_trust_impact", 0.0),
            "total_affected_rows": stats.get("total_affected_rows", 0),
            "reliability_rating": self._get_reliability_rating(column_name),
            "issue_patterns": self._analyze_issue_patterns(column_name),
            "recommendations": self._generate_recommendations(column_name)
        }
        
        return insights
    
    def _get_most_common_reason(self, column_name: str) -> str:
        """Get the most common revaluation reason for a column."""
        history = self.revaluation_history[column_name]
        if not history:
            return "none"
        
        reason_counts = {}
        for record in history:
            reason = record["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        return max(reason_counts.items(), key=lambda x: x[1])[0]
    
    def _get_reliability_rating(self, column_name: str) -> str:
        """Get reliability rating for a column."""
        confidence = self.confidence_scores[column_name]
        
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _analyze_issue_patterns(self, column_name: str) -> Dict[str, Any]:
        """Analyze issue patterns for a column."""
        history = self.revaluation_history[column_name]
        
        if not history:
            return {"patterns": [], "trend": "stable"}
        
        # Analyze temporal patterns
        timestamps = [datetime.fromisoformat(record["timestamp"]) for record in history]
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                     for i in range(1, len(timestamps))]
        
        # Analyze reason patterns
        reasons = [record["reason"] for record in history]
        reason_sequence = []
        for i in range(1, len(reasons)):
            reason_sequence.append(f"{reasons[i-1]} -> {reasons[i]}")
        
        patterns = {
            "total_issues": len(history),
            "avg_time_between_issues": np.mean(time_diffs) if time_diffs else 0,
            "issue_acceleration": self._calculate_acceleration(time_diffs),
            "common_transitions": self._get_common_transitions(reason_sequence),
            "seasonal_patterns": self._detect_seasonal_patterns(timestamps)
        }
        
        return patterns
    
    def _calculate_acceleration(self, time_diffs: List[float]) -> str:
        """Calculate if issues are accelerating or decelerating."""
        if len(time_diffs) < 3:
            return "insufficient_data"
        
        # Calculate trend
        early_avg = np.mean(time_diffs[:len(time_diffs)//2])
        late_avg = np.mean(time_diffs[len(time_diffs)//2:])
        
        if late_avg < early_avg * 0.7:
            return "accelerating"
        elif late_avg > early_avg * 1.3:
            return "decelerating"
        else:
            return "stable"
    
    def _get_common_transitions(self, transitions: List[str]) -> Dict[str, int]:
        """Get common reason transitions."""
        transition_counts = {}
        for transition in transitions:
            transition_counts[transition] = transition_counts.get(transition, 0) + 1
        
        return dict(sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    def _detect_seasonal_patterns(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Detect seasonal patterns in revaluations."""
        if len(timestamps) < 10:
            return {"has_patterns": False}
        
        # Analyze by hour of day
        hours = [ts.hour for ts in timestamps]
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Find peak hours
        peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "has_patterns": len(set(hours)) < len(hours) * 0.5,  # If hours are concentrated
            "peak_hours": peak_hours,
            "hour_distribution": hour_counts
        }
    
    def _generate_recommendations(self, column_name: str) -> List[str]:
        """Generate recommendations for improving column quality."""
        recommendations = []
        confidence = self.confidence_scores[column_name]
        history = self.revaluation_history[column_name]
        
        if confidence < 0.5:
            recommendations.append("Consider removing this column due to low reliability")
        
        if len(history) > 10:
            recommendations.append("High revaluation frequency - implement stricter validation")
        
        # Analyze specific issues
        reasons = [record["reason"] for record in history]
        if "outlier_detected" in reasons:
            recommendations.append("Implement outlier detection and handling")
        
        if "missing_values" in reasons:
            recommendations.append("Add missing value imputation strategy")
        
        if "inconsistency_found" in reasons:
            recommendations.append("Implement data consistency checks")
        
        if "logic_rule_failure" in reasons:
            recommendations.append("Review and update business logic rules")
        
        return recommendations
    
    def generate_column_visualization(self, save_path: str = None) -> str:
        """Generate column revaluation visualization."""
        if not self.revaluation_history or not any(self.revaluation_history.values()):
            return ""
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Revaluation frequency by column
        ax1 = axes[0, 0]
        columns = list(self.revaluation_history.keys())
        frequencies = [len(self.revaluation_history[col]) for col in columns]
        
        colors = ['red' if freq > 5 else 'orange' if freq > 2 else 'green' for freq in frequencies]
        bars = ax1.bar(columns, frequencies, color=colors, alpha=0.7)
        
        ax1.set_title('Revaluation Frequency by Column', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Columns')
        ax1.set_ylabel('Number of Revaluations')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, freq in zip(bars, frequencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(freq), ha='center', va='bottom', fontweight='bold')
        
        # Confidence scores
        ax2 = axes[0, 1]
        confidences = [self.confidence_scores[col] for col in columns]
        colors = ['green' if conf > 0.8 else 'orange' if conf > 0.6 else 'red' for conf in confidences]
        bars = ax2.bar(columns, confidences, color=colors, alpha=0.7)
        
        ax2.set_title('Column Confidence Scores', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Columns')
        ax2.set_ylabel('Confidence Score (0-1)')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, conf in zip(bars, confidences):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{conf:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Revaluation reasons heatmap
        ax3 = axes[1, 0]
        all_reasons = set()
        for col in columns:
            for record in self.revaluation_history[col]:
                all_reasons.add(record["reason"])
        
        reason_matrix = []
        for col in columns:
            col_reasons = [record["reason"] for record in self.revaluation_history[col]]
            row = [col_reasons.count(reason) for reason in all_reasons]
            reason_matrix.append(row)
        
        if reason_matrix:
            im = ax3.imshow(reason_matrix, cmap='YlOrRd', aspect='auto')
            ax3.set_title('Revaluation Reasons Heatmap', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Reasons')
            ax3.set_ylabel('Columns')
            ax3.set_xticks(range(len(all_reasons)))
            ax3.set_xticklabels(list(all_reasons), rotation=45, ha='right')
            ax3.set_yticks(range(len(columns)))
            ax3.set_yticklabels(columns)
            
            # Add colorbar
            plt.colorbar(im, ax=ax3, label='Count')
        
        # Timeline of revaluations
        ax4 = axes[1, 1]
        all_timestamps = []
        all_columns = []
        
        for col in columns:
            for record in self.revaluation_history[col]:
                all_timestamps.append(datetime.fromisoformat(record["timestamp"]))
                all_columns.append(col)
        
        if all_timestamps:
            ax4.scatter(all_timestamps, all_columns, alpha=0.6, s=50)
            ax4.set_title('Revaluation Timeline', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Columns')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.logs_dir / f"column_history_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def save_column_logs(self, filename: str = None) -> str:
        """Save column history logs to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"column_history_logs_{timestamp}.json"
        
        filepath = self.logs_dir / filename
        
        logs_data = {
            "column_names": self.column_names,
            "revaluation_history": self.revaluation_history,
            "column_stats": self.column_stats,
            "confidence_scores": self.confidence_scores,
            "total_revaluations": self.total_revaluations,
            "session_start": self.session_start.isoformat(),
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "session_duration": str(datetime.now() - self.session_start)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(logs_data, f, indent=2, default=str)
        
        return str(filepath)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get a summary report of column revaluation history."""
        # Calculate overall statistics
        total_columns = len(self.column_names)
        columns_with_issues = sum(1 for col in self.column_names 
                                if len(self.revaluation_history[col]) > 0)
        
        avg_confidence = np.mean(list(self.confidence_scores.values()))
        
        # Identify problematic columns
        problematic_columns = [col for col in self.column_names 
                             if self.confidence_scores[col] < 0.6]
        
        # Most common issues
        all_reasons = []
        for col in self.column_names:
            for record in self.revaluation_history[col]:
                all_reasons.append(record["reason"])
        
        reason_counts = {}
        for reason in all_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        most_common_issues = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_columns": total_columns,
            "columns_with_issues": columns_with_issues,
            "total_revaluations": self.total_revaluations,
            "average_confidence": float(avg_confidence),
            "problematic_columns": problematic_columns,
            "most_common_issues": most_common_issues,
            "session_duration": str(datetime.now() - self.session_start),
            "column_insights": {col: self.get_column_insights(col) for col in self.column_names}
        } 