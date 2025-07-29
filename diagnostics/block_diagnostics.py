#!/usr/bin/env python3
"""
Per-Block Diagnostic System for SREE
Provides detailed breakdown of which rows and features were reweighted or flagged in each block.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum


class DiagnosticAction(Enum):
    """Types of actions taken on rows/features."""
    DOWN_WEIGHTED = "down_weighted"
    RETAINED = "retained"
    FLAGGED = "flagged"
    REMOVED = "removed"
    ADJUSTED = "adjusted"
    NORMALIZED = "normalized"


class DiagnosticType(Enum):
    """Types of diagnostic checks."""
    ENTROPY = "entropy"
    HASH_CHANGE = "hash_change"
    LOGIC = "logic"
    PATTERN = "pattern"
    PRESENCE = "presence"
    PERMANENCE = "permanence"


@dataclass
class RowDiagnostic:
    """Diagnostic information for a specific row."""
    row_index: int
    original_weight: float
    final_weight: float
    weight_change: float
    diagnostic_type: DiagnosticType
    action_taken: DiagnosticAction
    reason: str
    feature_affected: Optional[str] = None
    rule_applied: Optional[str] = None
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "row_index": self.row_index,
            "original_weight": self.original_weight,
            "final_weight": self.final_weight,
            "weight_change": self.weight_change,
            "diagnostic_type": self.diagnostic_type.value,
            "action_taken": self.action_taken.value,
            "reason": self.reason,
            "feature_affected": self.feature_affected,
            "rule_applied": self.rule_applied,
            "confidence_score": self.confidence_score
        }


@dataclass
class FeatureDiagnostic:
    """Diagnostic information for a specific feature."""
    feature_name: str
    original_importance: float
    final_importance: float
    importance_change: float
    diagnostic_type: DiagnosticType
    action_taken: DiagnosticAction
    reason: str
    affected_rows: List[int] = field(default_factory=list)
    rule_applied: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_name": self.feature_name,
            "original_importance": self.original_importance,
            "final_importance": self.final_importance,
            "importance_change": self.importance_change,
            "diagnostic_type": self.diagnostic_type.value,
            "action_taken": self.action_taken.value,
            "reason": self.reason,
            "affected_rows": self.affected_rows,
            "rule_applied": self.rule_applied
        }


@dataclass
class BlockDiagnostic:
    """Complete diagnostic information for a single block."""
    block_number: int
    timestamp: str
    total_rows: int
    total_features: int
    rows_processed: int
    features_processed: int
    row_diagnostics: List[RowDiagnostic] = field(default_factory=list)
    feature_diagnostics: List[FeatureDiagnostic] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "block_number": self.block_number,
            "timestamp": self.timestamp,
            "total_rows": self.total_rows,
            "total_features": self.total_features,
            "rows_processed": self.rows_processed,
            "features_processed": self.features_processed,
            "row_diagnostics": [rd.to_dict() for rd in self.row_diagnostics],
            "feature_diagnostics": [fd.to_dict() for fd in self.feature_diagnostics],
            "summary_stats": self.summary_stats
        }


class BlockDiagnosticTracker:
    """
    Tracks and manages per-block diagnostics for SREE.
    """
    
    def __init__(self, enable_diagnostics: bool = True):
        """
        Initialize the diagnostic tracker.
        
        Args:
            enable_diagnostics: Whether to enable diagnostic tracking
        """
        self.enable_diagnostics = enable_diagnostics
        self.logger = logging.getLogger(__name__)
        self.block_diagnostics: List[BlockDiagnostic] = []
        self.current_block: Optional[BlockDiagnostic] = None
        
    def start_block(self, block_number: int, total_rows: int, total_features: int) -> None:
        """
        Start tracking diagnostics for a new block.
        
        Args:
            block_number: Current block number
            total_rows: Total number of rows in dataset
            total_features: Total number of features in dataset
        """
        if not self.enable_diagnostics:
            return
            
        self.current_block = BlockDiagnostic(
            block_number=block_number,
            timestamp=datetime.now().isoformat(),
            total_rows=total_rows,
            total_features=total_features,
            rows_processed=0,
            features_processed=0
        )
        
        self.logger.info(f"🔍 Starting diagnostics for Block {block_number}")
    
    def add_row_diagnostic(self, row_index: int, original_weight: float, final_weight: float,
                          diagnostic_type: DiagnosticType, action_taken: DiagnosticAction,
                          reason: str, feature_affected: Optional[str] = None,
                          rule_applied: Optional[str] = None, confidence_score: Optional[float] = None) -> None:
        """
        Add diagnostic information for a specific row.
        
        Args:
            row_index: Index of the row
            original_weight: Original weight of the row
            final_weight: Final weight after processing
            diagnostic_type: Type of diagnostic check
            action_taken: Action taken on the row
            reason: Reason for the action
            feature_affected: Feature that was affected (if any)
            rule_applied: Rule that was applied (if any)
            confidence_score: Confidence score for the action
        """
        if not self.enable_diagnostics or self.current_block is None:
            return
            
        weight_change = final_weight - original_weight
        
        row_diagnostic = RowDiagnostic(
            row_index=row_index,
            original_weight=original_weight,
            final_weight=final_weight,
            weight_change=weight_change,
            diagnostic_type=diagnostic_type,
            action_taken=action_taken,
            reason=reason,
            feature_affected=feature_affected,
            rule_applied=rule_applied,
            confidence_score=confidence_score
        )
        
        self.current_block.row_diagnostics.append(row_diagnostic)
        self.current_block.rows_processed += 1
        
        # Log significant changes
        if abs(weight_change) > 0.1:  # 10% change threshold
            self.logger.info(f"   Row {row_index}: {action_taken.value} ({weight_change:+.3f}) - {reason}")
    
    def add_feature_diagnostic(self, feature_name: str, original_importance: float, final_importance: float,
                              diagnostic_type: DiagnosticType, action_taken: DiagnosticAction,
                              reason: str, affected_rows: List[int] = None,
                              rule_applied: Optional[str] = None) -> None:
        """
        Add diagnostic information for a specific feature.
        
        Args:
            feature_name: Name of the feature
            original_importance: Original importance of the feature
            final_importance: Final importance after processing
            diagnostic_type: Type of diagnostic check
            action_taken: Action taken on the feature
            reason: Reason for the action
            affected_rows: List of row indices affected by this feature
            rule_applied: Rule that was applied (if any)
        """
        if not self.enable_diagnostics or self.current_block is None:
            return
            
        importance_change = final_importance - original_importance
        
        feature_diagnostic = FeatureDiagnostic(
            feature_name=feature_name,
            original_importance=original_importance,
            final_importance=final_importance,
            importance_change=importance_change,
            diagnostic_type=diagnostic_type,
            action_taken=action_taken,
            reason=reason,
            affected_rows=affected_rows or [],
            rule_applied=rule_applied
        )
        
        self.current_block.feature_diagnostics.append(feature_diagnostic)
        self.current_block.features_processed += 1
        
        # Log significant changes
        if abs(importance_change) > 0.05:  # 5% change threshold
            self.logger.info(f"   Feature '{feature_name}': {action_taken.value} ({importance_change:+.3f}) - {reason}")
    
    def end_block(self, summary_stats: Dict[str, Any] = None) -> None:
        """
        End tracking for the current block and save diagnostics.
        
        Args:
            summary_stats: Additional summary statistics for the block
        """
        if not self.enable_diagnostics or self.current_block is None:
            return
            
        # Add summary statistics
        if summary_stats:
            self.current_block.summary_stats = summary_stats
        
        # Calculate additional summary stats
        self._calculate_block_summary()
        
        # Save the block diagnostic
        self.block_diagnostics.append(self.current_block)
        
        # Log block summary
        self._log_block_summary()
        
        self.current_block = None
    
    def _calculate_block_summary(self) -> None:
        """Calculate summary statistics for the current block."""
        if self.current_block is None:
            return
            
        # Row-level summary
        row_actions = {}
        for rd in self.current_block.row_diagnostics:
            action = rd.action_taken.value
            row_actions[action] = row_actions.get(action, 0) + 1
        
        # Feature-level summary
        feature_actions = {}
        for fd in self.current_block.feature_diagnostics:
            action = fd.action_taken.value
            feature_actions[action] = feature_actions.get(action, 0) + 1
        
        # Weight changes summary
        weight_changes = [rd.weight_change for rd in self.current_block.row_diagnostics]
        importance_changes = [fd.importance_change for fd in self.current_block.feature_diagnostics]
        
        self.current_block.summary_stats.update({
            "row_actions": row_actions,
            "feature_actions": feature_actions,
            "avg_weight_change": np.mean(weight_changes) if weight_changes else 0.0,
            "avg_importance_change": np.mean(importance_changes) if importance_changes else 0.0,
            "total_weight_change": np.sum(weight_changes) if weight_changes else 0.0,
            "total_importance_change": np.sum(importance_changes) if importance_changes else 0.0
        })
    
    def _log_block_summary(self) -> None:
        """Log a summary of the current block."""
        if self.current_block is None:
            return
            
        block = self.current_block
        
        self.logger.info(f"📊 Block {block.block_number} Summary:")
        self.logger.info(f"   Rows processed: {block.rows_processed}/{block.total_rows}")
        self.logger.info(f"   Features processed: {block.features_processed}/{block.total_features}")
        
        # Log row actions
        if block.summary_stats.get("row_actions"):
            self.logger.info(f"   Row actions: {block.summary_stats['row_actions']}")
        
        # Log feature actions
        if block.summary_stats.get("feature_actions"):
            self.logger.info(f"   Feature actions: {block.summary_stats['feature_actions']}")
        
        # Log weight changes
        avg_weight_change = block.summary_stats.get("avg_weight_change", 0.0)
        total_weight_change = block.summary_stats.get("total_weight_change", 0.0)
        self.logger.info(f"   Weight changes: avg={avg_weight_change:+.3f}, total={total_weight_change:+.3f}")
    
    def get_all_diagnostics(self) -> List[Dict[str, Any]]:
        """
        Get all block diagnostics as a list of dictionaries.
        
        Returns:
            List of block diagnostic dictionaries
        """
        return [bd.to_dict() for bd in self.block_diagnostics]
    
    def get_block_diagnostic(self, block_number: int) -> Optional[Dict[str, Any]]:
        """
        Get diagnostic information for a specific block.
        
        Args:
            block_number: Block number to retrieve
            
        Returns:
            Block diagnostic dictionary or None if not found
        """
        for bd in self.block_diagnostics:
            if bd.block_number == block_number:
                return bd.to_dict()
        return None
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Get a summary report of all diagnostics.
        
        Returns:
            Summary report dictionary
        """
        if not self.block_diagnostics:
            return {"message": "No diagnostics available"}
        
        total_blocks = len(self.block_diagnostics)
        total_rows_processed = sum(bd.rows_processed for bd in self.block_diagnostics)
        total_features_processed = sum(bd.features_processed for bd in self.block_diagnostics)
        
        # Aggregate actions across all blocks
        all_row_actions = {}
        all_feature_actions = {}
        
        for bd in self.block_diagnostics:
            for action, count in bd.summary_stats.get("row_actions", {}).items():
                all_row_actions[action] = all_row_actions.get(action, 0) + count
            
            for action, count in bd.summary_stats.get("feature_actions", {}).items():
                all_feature_actions[action] = all_feature_actions.get(action, 0) + count
        
        return {
            "total_blocks": total_blocks,
            "total_rows_processed": total_rows_processed,
            "total_features_processed": total_features_processed,
            "all_row_actions": all_row_actions,
            "all_feature_actions": all_feature_actions,
            "block_details": [bd.to_dict() for bd in self.block_diagnostics]
        }


# Global diagnostic tracker instance
diagnostic_tracker = BlockDiagnosticTracker()


def enable_diagnostics(enable: bool = True) -> None:
    """
    Enable or disable diagnostic tracking globally.
    
    Args:
        enable: Whether to enable diagnostics
    """
    global diagnostic_tracker
    diagnostic_tracker.enable_diagnostics = enable
    if enable:
        logging.getLogger(__name__).info("🔍 Diagnostics enabled")
    else:
        logging.getLogger(__name__).info("🔍 Diagnostics disabled")


def get_diagnostic_tracker() -> BlockDiagnosticTracker:
    """
    Get the global diagnostic tracker instance.
    
    Returns:
        Global diagnostic tracker
    """
    return diagnostic_tracker 