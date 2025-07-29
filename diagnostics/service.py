"""
SREE Per-Block Diagnostics Service
Core service for managing diagnostic records and summaries.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import numpy as np

from .diagnostic_types import (
    BlockRowDiagnostic,
    BlockDiagnosticSummary,
    DiagnosticsConfig,
    DiagnosticAction,
    ColumnInsight
)


class DiagnosticsService:
    """
    Service for managing per-block diagnostics with feature flagging.
    
    This service provides transparent, auditable logs per block that show:
    - Which rows had low V_q (entropy), V_b (hash change), or V_l (logic)
    - What action was taken on each row (e.g., down-weighted, retained, flagged)
    - Weight deltas and column-level insights
    """
    
    def __init__(self, config: Optional[DiagnosticsConfig] = None):
        """
        Initialize diagnostics service.
        
        Args:
            config: Configuration for diagnostics (defaults to disabled)
        """
        self.config = config or DiagnosticsConfig()
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage
        self._diagnostics: Dict[str, List[BlockRowDiagnostic]] = {}
        self._summaries: Dict[str, List[BlockDiagnosticSummary]] = {}
        
        # Setup persistence directory if enabled
        if self.config.persist:
            self._setup_persistence()
    
    def _setup_persistence(self):
        """Setup persistence directory for diagnostics."""
        try:
            self.persistence_dir = Path("logs/diagnostics")
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Diagnostics persistence enabled: {self.persistence_dir}")
        except Exception as e:
            self.logger.error(f"Failed to setup persistence: {e}")
            self.config.persist = False
    
    def emit_row(self, run_id: str, block_index: int, payload: Dict) -> None:
        """
        Emit a diagnostic record for a single row.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the current block
            payload: Diagnostic data (excluding timestamp which is auto-generated)
        """
        if not self.config.enabled:
            return  # No-op when disabled
        
        try:
            # Create diagnostic record
            diagnostic = BlockRowDiagnostic(
                run_id=run_id,
                block_index=block_index,
                row_id=payload['row_id'],
                action=payload['action'],
                vq=payload.get('vq'),
                vb=payload.get('vb'),
                vl=payload.get('vl'),
                weight_delta=payload.get('weight_delta'),
                columns=payload.get('columns'),
                reason=payload.get('reason')
            )
            
            # Store in memory
            if run_id not in self._diagnostics:
                self._diagnostics[run_id] = []
            
            # Check max rows limit
            if len(self._diagnostics[run_id]) >= self.config.max_rows_per_block:
                self.logger.warning(
                    f"Max rows per block ({self.config.max_rows_per_block}) exceeded for run {run_id}. "
                    "Sampling diagnostics to prevent memory explosion."
                )
                # Sample by keeping every nth row
                sample_rate = len(self._diagnostics[run_id]) // self.config.max_rows_per_block + 1
                if len(self._diagnostics[run_id]) % sample_rate == 0:
                    self._diagnostics[run_id].append(diagnostic)
            else:
                self._diagnostics[run_id].append(diagnostic)
            
            # Persist if enabled
            if self.config.persist:
                self._persist_diagnostic(diagnostic)
                
        except Exception as e:
            self.logger.error(f"Failed to emit diagnostic: {e}")
    
    def _persist_diagnostic(self, diagnostic: BlockRowDiagnostic):
        """Persist a diagnostic record to disk."""
        try:
            # Create run-specific directory
            run_dir = self.persistence_dir / diagnostic.run_id
            run_dir.mkdir(exist_ok=True)
            
            # Append to block-specific file
            block_file = run_dir / f"block_{diagnostic.block_index}.jsonl"
            
            with open(block_file, 'a') as f:
                f.write(json.dumps(diagnostic.to_dict()) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to persist diagnostic: {e}")
    
    def summarize_block(self, run_id: str, block_index: int) -> BlockDiagnosticSummary:
        """
        Create a summary for a specific block.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the block to summarize
            
        Returns:
            BlockDiagnosticSummary for the specified block
        """
        if not self.config.enabled:
            # Return empty summary when disabled
            return BlockDiagnosticSummary(
                run_id=run_id,
                block_index=block_index,
                affected_rows=0
            )
        
        # Get diagnostics for this block
        block_diagnostics = [
            d for d in self._diagnostics.get(run_id, [])
            if d.block_index == block_index
        ]
        
        if not block_diagnostics:
            return BlockDiagnosticSummary(
                run_id=run_id,
                block_index=block_index,
                affected_rows=0
            )
        
        # Calculate averages
        vq_values = [d.vq for d in block_diagnostics if d.vq is not None]
        vb_values = [d.vb for d in block_diagnostics if d.vb is not None]
        vl_values = [d.vl for d in block_diagnostics if d.vl is not None]
        
        # Count actions
        actions_count = {'DOWN_WEIGHTED': 0, 'RETAINED': 0, 'FLAGGED': 0}
        for diagnostic in block_diagnostics:
            actions_count[diagnostic.action] += 1
        
        # Create summary
        summary = BlockDiagnosticSummary(
            run_id=run_id,
            block_index=block_index,
            affected_rows=len(block_diagnostics),
            avg_vq=np.mean(vq_values) if vq_values else None,
            avg_vb=np.mean(vb_values) if vb_values else None,
            avg_vl=np.mean(vl_values) if vl_values else None,
            actions_count=actions_count
        )
        
        # Store summary
        if run_id not in self._summaries:
            self._summaries[run_id] = []
        self._summaries[run_id].append(summary)
        
        return summary
    
    def get_block_diagnostics(self, run_id: str, block_index: int) -> List[BlockRowDiagnostic]:
        """
        Get all diagnostics for a specific block.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the block
            
        Returns:
            List of BlockRowDiagnostic records for the block
        """
        if not self.config.enabled:
            return []
        
        # Return from memory
        return [
            d for d in self._diagnostics.get(run_id, [])
            if d.block_index == block_index
        ]
    
    def get_block_summaries(self, run_id: str) -> List[BlockDiagnosticSummary]:
        """
        Get all block summaries for a run.
        
        Args:
            run_id: Unique identifier for the analysis run
            
        Returns:
            List of BlockDiagnosticSummary records for the run
        """
        if not self.config.enabled:
            return []
        
        return self._summaries.get(run_id, [])
    
    def export_csv(self, run_id: str, output_path: Optional[str] = None) -> str:
        """
        Export diagnostics to CSV format.
        
        Args:
            run_id: Unique identifier for the analysis run
            output_path: Optional output path (defaults to logs/diagnostics/{run_id}.csv)
            
        Returns:
            Path to the exported CSV file
        """
        if not self.config.enabled:
            raise ValueError("Diagnostics must be enabled to export data")
        
        try:
            import pandas as pd
            
            # Get all diagnostics for the run
            diagnostics = self._diagnostics.get(run_id, [])
            
            if not diagnostics:
                raise ValueError(f"No diagnostics found for run {run_id}")
            
            # Convert to DataFrame
            data = []
            for diagnostic in diagnostics:
                row_data = {
                    'run_id': diagnostic.run_id,
                    'block_index': diagnostic.block_index,
                    'row_id': diagnostic.row_id,
                    'action': diagnostic.action,
                    'timestamp': diagnostic.timestamp,
                    'vq': diagnostic.vq,
                    'vb': diagnostic.vb,
                    'vl': diagnostic.vl,
                    'weight_delta': diagnostic.weight_delta,
                    'reason': diagnostic.reason
                }
                
                # Add column insights
                if diagnostic.columns:
                    for i, col in enumerate(diagnostic.columns):
                        row_data[f'column_{i+1}'] = col.column
                        row_data[f'rule_{i+1}'] = col.rule
                        row_data[f'delta_{i+1}'] = col.delta
                        row_data[f'reason_{i+1}'] = col.reason
                
                data.append(row_data)
            
            df = pd.DataFrame(data)
            
            # Determine output path
            if output_path is None:
                output_path = f"logs/diagnostics/{run_id}_diagnostics.csv"
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Diagnostics exported to {output_path}")
            return output_path
            
        except ImportError:
            raise ImportError("pandas is required for CSV export")
        except Exception as e:
            self.logger.error(f"Failed to export diagnostics: {e}")
            raise
    
    def clear_run(self, run_id: str):
        """
        Clear diagnostics for a specific run.
        
        Args:
            run_id: Unique identifier for the analysis run
        """
        if run_id in self._diagnostics:
            del self._diagnostics[run_id]
        if run_id in self._summaries:
            del self._summaries[run_id]
        
        # Clear persisted files if enabled
        if self.config.persist:
            try:
                run_dir = self.persistence_dir / run_id
                if run_dir.exists():
                    import shutil
                    shutil.rmtree(run_dir)
            except Exception as e:
                self.logger.error(f"Failed to clear persisted diagnostics: {e}")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about stored diagnostics.
        
        Returns:
            Dictionary with diagnostic statistics
        """
        if not self.config.enabled:
            return {"enabled": False}
        
        total_runs = len(self._diagnostics)
        total_diagnostics = sum(len(diags) for diags in self._diagnostics.values())
        total_summaries = sum(len(sums) for sums in self._summaries.values())
        
        return {
            "enabled": True,
            "total_runs": total_runs,
            "total_diagnostics": total_diagnostics,
            "total_summaries": total_summaries,
            "persistence_enabled": self.config.persist,
            "max_rows_per_block": self.config.max_rows_per_block
        }


# Global instance for easy access
_diagnostics_service: Optional[DiagnosticsService] = None


def get_diagnostics_service() -> DiagnosticsService:
    """Get the global diagnostics service instance."""
    global _diagnostics_service
    if _diagnostics_service is None:
        _diagnostics_service = DiagnosticsService()
    return _diagnostics_service


def set_diagnostics_service(service: DiagnosticsService):
    """Set the global diagnostics service instance."""
    global _diagnostics_service
    _diagnostics_service = service 