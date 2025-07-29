"""
SREE Per-Block Diagnostics Hooks
Integration hooks for the block refinement loop.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np

from .service import get_diagnostics_service
from .diagnostic_types import DiagnosticAction, ColumnInsight


class DiagnosticsHooks:
    """
    Hooks for integrating diagnostics into the SREE block refinement loop.
    
    This class provides methods that can be called at various points in the
    block refinement process to emit diagnostic information without breaking
    existing functionality.
    """
    
    def __init__(self):
        """Initialize diagnostics hooks."""
        self.service = get_diagnostics_service()
        self.logger = logging.getLogger(__name__)
    
    def emit_block_start(self, run_id: str, block_index: int, block_size: int):
        """
        Emit diagnostic information when a block starts processing.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the current block
            block_size: Number of rows in the block
        """
        if not self.service.config.enabled:
            return
        
        self.logger.debug(f"Block {block_index} started processing {block_size} rows for run {run_id}")
    
    def emit_row_validation(
        self,
        run_id: str,
        block_index: int,
        row_id: Union[str, int],
        vq: Optional[float] = None,
        vb: Optional[float] = None,
        vl: Optional[float] = None,
        action: DiagnosticAction = 'RETAINED',
        weight_delta: Optional[float] = None,
        columns: Optional[List[ColumnInsight]] = None,
        reason: Optional[str] = None
    ):
        """
        Emit diagnostic information for a single row validation.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the current block
            row_id: Identifier for the row
            vq: Entropy validation score
            vb: Hash/change validation score
            vl: Logic validation score
            action: Action taken on the row
            weight_delta: Weight change applied
            columns: Column-level insights
            reason: Explanation for the action
        """
        if not self.service.config.enabled:
            return
        
        payload = {
            'row_id': row_id,
            'action': action,
            'vq': vq,
            'vb': vb,
            'vl': vl,
            'weight_delta': weight_delta,
            'columns': columns,
            'reason': reason
        }
        
        self.service.emit_row(run_id, block_index, payload)
    
    def emit_block_end(self, run_id: str, block_index: int) -> Optional[Dict]:
        """
        Emit diagnostic information when a block finishes processing.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the current block
            
        Returns:
            Block summary if diagnostics are enabled, None otherwise
        """
        if not self.service.config.enabled:
            return None
        
        summary = self.service.summarize_block(run_id, block_index)
        self.logger.debug(f"Block {block_index} finished processing for run {run_id}: {summary.affected_rows} affected rows")
        
        return summary.to_dict()
    
    def emit_logic_validation(
        self,
        run_id: str,
        block_index: int,
        row_id: Union[str, int],
        feature_names: List[str],
        feature_values: np.ndarray,
        logic_rules: List[Dict],
        validation_results: List[bool],
        weight_delta: float
    ) -> List[ColumnInsight]:
        """
        Emit detailed logic validation diagnostics.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the current block
            row_id: Identifier for the row
            feature_names: Names of the features
            feature_values: Values of the features for this row
            logic_rules: Logic rules that were applied
            validation_results: Results of each rule validation
            weight_delta: Weight change applied due to logic validation
            
        Returns:
            List of column insights for the row
        """
        if not self.service.config.enabled:
            return []
        
        column_insights = []
        
        for i, (rule, result) in enumerate(zip(logic_rules, validation_results)):
            if not result:  # Rule failed
                feature_idx = rule.get('feature_index', i)
                if feature_idx < len(feature_names):
                    column_name = feature_names[feature_idx]
                    feature_value = feature_values[feature_idx] if feature_idx < len(feature_values) else None
                    
                    # Create rule description
                    rule_desc = rule.get('description', f"Rule {i+1}")
                    if feature_value is not None:
                        rule_desc += f" (value: {feature_value:.3f})"
                    
                    insight = ColumnInsight(
                        column=column_name,
                        rule=rule_desc,
                        delta=weight_delta,
                        reason=f"Logic validation failed: {rule.get('reason', 'Rule violation')}"
                    )
                    column_insights.append(insight)
        
        return column_insights
    
    def emit_entropy_validation(
        self,
        run_id: str,
        block_index: int,
        row_id: Union[str, int],
        entropy_score: float,
        threshold: float,
        weight_delta: float
    ):
        """
        Emit entropy validation diagnostics.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the current block
            row_id: Identifier for the row
            entropy_score: Calculated entropy score
            threshold: Entropy threshold
            weight_delta: Weight change applied
        """
        if not self.service.config.enabled:
            return
        
        action = 'DOWN_WEIGHTED' if entropy_score > threshold else 'RETAINED'
        reason = f"Entropy {entropy_score:.3f} {'exceeds' if entropy_score > threshold else 'within'} threshold {threshold:.3f}"
        
        self.emit_row_validation(
            run_id=run_id,
            block_index=block_index,
            row_id=row_id,
            vq=entropy_score,
            action=action,
            weight_delta=weight_delta,
            reason=reason
        )
    
    def emit_hash_validation(
        self,
        run_id: str,
        block_index: int,
        row_id: Union[str, int],
        hash_score: float,
        threshold: float,
        weight_delta: float
    ):
        """
        Emit hash/change validation diagnostics.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the current block
            row_id: Identifier for the row
            hash_score: Calculated hash/change score
            threshold: Hash threshold
            weight_delta: Weight change applied
        """
        if not self.service.config.enabled:
            return
        
        action = 'DOWN_WEIGHTED' if hash_score > threshold else 'RETAINED'
        reason = f"Hash change {hash_score:.3f} {'exceeds' if hash_score > threshold else 'within'} threshold {threshold:.3f}"
        
        self.emit_row_validation(
            run_id=run_id,
            block_index=block_index,
            row_id=row_id,
            vb=hash_score,
            action=action,
            weight_delta=weight_delta,
            reason=reason
        )
    
    def emit_flagged_row(
        self,
        run_id: str,
        block_index: int,
        row_id: Union[str, int],
        reason: str,
        vq: Optional[float] = None,
        vb: Optional[float] = None,
        vl: Optional[float] = None
    ):
        """
        Emit diagnostic information for a flagged row.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the current block
            row_id: Identifier for the row
            reason: Reason for flagging
            vq: Entropy validation score
            vb: Hash/change validation score
            vl: Logic validation score
        """
        if not self.service.config.enabled:
            return
        
        self.emit_row_validation(
            run_id=run_id,
            block_index=block_index,
            row_id=row_id,
            vq=vq,
            vb=vb,
            vl=vl,
            action='FLAGGED',
            reason=reason
        )


# Global instance for easy access
_diagnostics_hooks: Optional[DiagnosticsHooks] = None


def get_diagnostics_hooks() -> DiagnosticsHooks:
    """Get the global diagnostics hooks instance."""
    global _diagnostics_hooks
    if _diagnostics_hooks is None:
        _diagnostics_hooks = DiagnosticsHooks()
    return _diagnostics_hooks 