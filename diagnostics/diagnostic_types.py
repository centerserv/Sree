"""
SREE Per-Block Diagnostics Types
Type definitions for diagnostic records and summaries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal
from datetime import datetime
import json


# Type aliases for better readability
DiagnosticAction = Literal['DOWN_WEIGHTED', 'RETAINED', 'FLAGGED']
RowId = Union[str, int]


@dataclass
class ColumnInsight:
    """Insight about a specific column/feature that affected a row."""
    column: str
    rule: Optional[str] = None  # e.g. "cholesterol > 240"
    delta: Optional[float] = None  # e.g. -0.25
    reason: Optional[str] = None  # free text explanation
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'column': self.column,
            'rule': self.rule,
            'delta': self.delta,
            'reason': self.reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ColumnInsight':
        """Create from dictionary."""
        return cls(
            column=data['column'],
            rule=data.get('rule'),
            delta=data.get('delta'),
            reason=data.get('reason')
        )


@dataclass
class BlockRowDiagnostic:
    """Diagnostic record for a single row in a block."""
    run_id: str
    block_index: int
    row_id: RowId
    action: DiagnosticAction
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Validation scores (optional)
    vq: Optional[float] = None  # entropy
    vb: Optional[float] = None  # hash/change score
    vl: Optional[float] = None  # logic score
    
    # Action details
    weight_delta: Optional[float] = None  # e.g. -0.25
    columns: Optional[List[ColumnInsight]] = None
    reason: Optional[str] = None  # short explanation for auditors
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'block_index': self.block_index,
            'row_id': self.row_id,
            'action': self.action,
            'timestamp': self.timestamp,
            'vq': self.vq,
            'vb': self.vb,
            'vl': self.vl,
            'weight_delta': self.weight_delta,
            'columns': [col.to_dict() for col in (self.columns or [])],
            'reason': self.reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BlockRowDiagnostic':
        """Create from dictionary."""
        return cls(
            run_id=data['run_id'],
            block_index=data['block_index'],
            row_id=data['row_id'],
            action=data['action'],
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            vq=data.get('vq'),
            vb=data.get('vb'),
            vl=data.get('vl'),
            weight_delta=data.get('weight_delta'),
            columns=[ColumnInsight.from_dict(col) for col in data.get('columns', [])],
            reason=data.get('reason')
        )


@dataclass
class BlockDiagnosticSummary:
    """Aggregated summary for a block's diagnostics."""
    run_id: str
    block_index: int
    affected_rows: int = 0
    
    # Average validation scores
    avg_vq: Optional[float] = None
    avg_vb: Optional[float] = None
    avg_vl: Optional[float] = None
    
    # Action counts
    actions_count: Dict[DiagnosticAction, int] = field(
        default_factory=lambda: {'DOWN_WEIGHTED': 0, 'RETAINED': 0, 'FLAGGED': 0}
    )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'block_index': self.block_index,
            'affected_rows': self.affected_rows,
            'avg_vq': self.avg_vq,
            'avg_vb': self.avg_vb,
            'avg_vl': self.avg_vl,
            'actions_count': self.actions_count.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BlockDiagnosticSummary':
        """Create from dictionary."""
        return cls(
            run_id=data['run_id'],
            block_index=data['block_index'],
            affected_rows=data.get('affected_rows', 0),
            avg_vq=data.get('avg_vq'),
            avg_vb=data.get('avg_vb'),
            avg_vl=data.get('avg_vl'),
            actions_count=data.get('actions_count', {'DOWN_WEIGHTED': 0, 'RETAINED': 0, 'FLAGGED': 0})
        )


# Configuration types
@dataclass
class DiagnosticsConfig:
    """Configuration for per-block diagnostics."""
    enabled: bool = False
    persist: bool = False
    max_rows_per_block: int = 10000
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'enabled': self.enabled,
            'persist': self.persist,
            'max_rows_per_block': self.max_rows_per_block
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DiagnosticsConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', False),
            persist=data.get('persist', False),
            max_rows_per_block=data.get('max_rows_per_block', 10000)
        ) 