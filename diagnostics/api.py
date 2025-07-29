"""
SREE Per-Block Diagnostics API
REST API endpoints for accessing diagnostic data.
"""

import json
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

from .service import get_diagnostics_service
from .diagnostic_types import BlockRowDiagnostic, BlockDiagnosticSummary


class DiagnosticsAPI:
    """
    REST API for accessing SREE per-block diagnostics.
    
    This class provides endpoints for retrieving diagnostic data
    without modifying existing functionality.
    """
    
    def __init__(self):
        """Initialize diagnostics API."""
        self.service = get_diagnostics_service()
        self.logger = logging.getLogger(__name__)
    
    def get_block_summaries(self, run_id: str) -> List[Dict]:
        """
        Get all block summaries for a run.
        
        Args:
            run_id: Unique identifier for the analysis run
            
        Returns:
            List of block summary dictionaries
        """
        try:
            summaries = self.service.get_block_summaries(run_id)
            return [summary.to_dict() for summary in summaries]
        except Exception as e:
            self.logger.error(f"Failed to get block summaries for run {run_id}: {e}")
            return []
    
    def get_block_diagnostics(self, run_id: str, block_index: int) -> List[Dict]:
        """
        Get all diagnostics for a specific block.
        
        Args:
            run_id: Unique identifier for the analysis run
            block_index: Index of the block
            
        Returns:
            List of diagnostic record dictionaries
        """
        try:
            diagnostics = self.service.get_block_diagnostics(run_id, block_index)
            return [diagnostic.to_dict() for diagnostic in diagnostics]
        except Exception as e:
            self.logger.error(f"Failed to get block diagnostics for run {run_id}, block {block_index}: {e}")
            return []
    
    def export_csv(self, run_id: str, output_path: Optional[str] = None) -> Dict:
        """
        Export diagnostics to CSV format.
        
        Args:
            run_id: Unique identifier for the analysis run
            output_path: Optional output path
            
        Returns:
            Dictionary with export information
        """
        try:
            csv_path = self.service.export_csv(run_id, output_path)
            return {
                "success": True,
                "file_path": csv_path,
                "run_id": run_id,
                "exported_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to export CSV for run {run_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "run_id": run_id
            }
    
    def get_stats(self) -> Dict:
        """
        Get statistics about stored diagnostics.
        
        Returns:
            Dictionary with diagnostic statistics
        """
        return self.service.get_stats()
    
    def clear_run(self, run_id: str) -> Dict:
        """
        Clear diagnostics for a specific run.
        
        Args:
            run_id: Unique identifier for the analysis run
            
        Returns:
            Dictionary with operation result
        """
        try:
            self.service.clear_run(run_id)
            return {
                "success": True,
                "run_id": run_id,
                "message": f"Diagnostics cleared for run {run_id}"
            }
        except Exception as e:
            self.logger.error(f"Failed to clear diagnostics for run {run_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "run_id": run_id
            }
    
    def get_available_runs(self) -> List[str]:
        """
        Get list of available run IDs.
        
        Returns:
            List of run IDs that have diagnostics data
        """
        try:
            stats = self.service.get_stats()
            if not stats.get("enabled", False):
                return []
            
            # This would need to be implemented in the service
            # For now, return empty list
            return []
        except Exception as e:
            self.logger.error(f"Failed to get available runs: {e}")
            return []


# Global instance for easy access
_diagnostics_api: Optional[DiagnosticsAPI] = None


def get_diagnostics_api() -> DiagnosticsAPI:
    """Get the global diagnostics API instance."""
    global _diagnostics_api
    if _diagnostics_api is None:
        _diagnostics_api = DiagnosticsAPI()
    return _diagnostics_api


# Flask/Streamlit integration helpers
def create_diagnostics_routes(app):
    """
    Create Flask routes for diagnostics API.
    
    Args:
        app: Flask application instance
    """
    api = get_diagnostics_api()
    
    @app.route('/sree/diagnostics/<run_id>/blocks', methods=['GET'])
    def get_block_summaries_route(run_id):
        """GET /sree/diagnostics/:runId/blocks → BlockDiagnosticSummary[]"""
        try:
            summaries = api.get_block_summaries(run_id)
            return json.dumps(summaries), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            return json.dumps({"error": str(e)}), 500, {'Content-Type': 'application/json'}
    
    @app.route('/sree/diagnostics/<run_id>/block/<int:block_index>', methods=['GET'])
    def get_block_diagnostics_route(run_id, block_index):
        """GET /sree/diagnostics/:runId/block/:index → BlockRowDiagnostic[]"""
        try:
            diagnostics = api.get_block_diagnostics(run_id, block_index)
            return json.dumps(diagnostics), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            return json.dumps({"error": str(e)}), 500, {'Content-Type': 'application/json'}
    
    @app.route('/sree/diagnostics/<run_id>/export.csv', methods=['GET'])
    def export_csv_route(run_id):
        """GET /sree/diagnostics/:runId/export.csv"""
        try:
            result = api.export_csv(run_id)
            if result["success"]:
                return json.dumps(result), 200, {'Content-Type': 'application/json'}
            else:
                return json.dumps(result), 400, {'Content-Type': 'application/json'}
        except Exception as e:
            return json.dumps({"error": str(e)}), 500, {'Content-Type': 'application/json'}
    
    @app.route('/sree/diagnostics/stats', methods=['GET'])
    def get_stats_route():
        """GET /sree/diagnostics/stats"""
        try:
            stats = api.get_stats()
            return json.dumps(stats), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            return json.dumps({"error": str(e)}), 500, {'Content-Type': 'application/json'}
    
    @app.route('/sree/diagnostics/<run_id>', methods=['DELETE'])
    def clear_run_route(run_id):
        """DELETE /sree/diagnostics/:runId"""
        try:
            result = api.clear_run(run_id)
            if result["success"]:
                return json.dumps(result), 200, {'Content-Type': 'application/json'}
            else:
                return json.dumps(result), 400, {'Content-Type': 'application/json'}
        except Exception as e:
            return json.dumps({"error": str(e)}), 500, {'Content-Type': 'application/json'}


def create_streamlit_diagnostics_section():
    """
    Create Streamlit section for diagnostics.
    
    Returns:
        Function that renders the diagnostics section
    """
    def render_diagnostics_section():
        import streamlit as st
        api = get_diagnostics_api()
        
        st.header("🔍 Per-Block Diagnostics")
        
        # Check if diagnostics are enabled
        stats = api.get_stats()
        if not stats.get("enabled", False):
            st.info("Per-block diagnostics are currently disabled. Enable them in the configuration to see detailed diagnostics.")
            return
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Runs", stats.get("total_runs", 0))
        with col2:
            st.metric("Total Diagnostics", stats.get("total_diagnostics", 0))
        with col3:
            st.metric("Total Summaries", stats.get("total_summaries", 0))
        
        # Run selection
        st.subheader("📊 View Diagnostics")
        
        # For now, show a placeholder since we don't have run selection implemented
        st.info("Run selection will be available when diagnostics are generated during analysis.")
        
        # Export section
        st.subheader("📥 Export Diagnostics")
        if st.button("Export All Diagnostics (CSV)"):
            st.info("Export functionality will be available when diagnostics are generated.")
    
    return render_diagnostics_section 