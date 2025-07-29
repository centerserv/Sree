"""
SREE Per-Block Diagnostics Module
Provides transparent, auditable logs per block for SREE analysis.
"""

from .diagnostic_types import (
    DiagnosticAction,
    ColumnInsight,
    BlockRowDiagnostic,
    BlockDiagnosticSummary,
    DiagnosticsConfig
)
from .service import DiagnosticsService, get_diagnostics_service, set_diagnostics_service
from .hooks import DiagnosticsHooks, get_diagnostics_hooks
from .api import DiagnosticsAPI, get_diagnostics_api, create_diagnostics_routes, create_streamlit_diagnostics_section

__all__ = [
    'DiagnosticAction',
    'ColumnInsight', 
    'BlockRowDiagnostic',
    'BlockDiagnosticSummary',
    'DiagnosticsConfig',
    'DiagnosticsService',
    'get_diagnostics_service',
    'set_diagnostics_service',
    'DiagnosticsHooks',
    'get_diagnostics_hooks',
    'DiagnosticsAPI',
    'get_diagnostics_api',
    'create_diagnostics_routes',
    'create_streamlit_diagnostics_section'
] 