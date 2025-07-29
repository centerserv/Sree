"""
Tests for SREE Per-Block Diagnostics System
Comprehensive test suite covering all diagnostic functionality.
"""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import diagnostics components
from diagnostics import (
    DiagnosticsService,
    DiagnosticsConfig,
    BlockRowDiagnostic,
    BlockDiagnosticSummary,
    ColumnInsight,
    DiagnosticAction,
    get_diagnostics_service,
    get_diagnostics_hooks,
    DiagnosticsAPI
)


class TestDiagnosticsTypes:
    """Test diagnostic data types."""
    
    def test_column_insight_creation(self):
        """Test ColumnInsight creation and serialization."""
        insight = ColumnInsight(
            column="cholesterol",
            rule="cholesterol > 240",
            delta=-0.25,
            reason="Logic validation failed: Rule violation"
        )
        
        assert insight.column == "cholesterol"
        assert insight.rule == "cholesterol > 240"
        assert insight.delta == -0.25
        assert insight.reason == "Logic validation failed: Rule violation"
        
        # Test serialization
        data = insight.to_dict()
        assert data["column"] == "cholesterol"
        assert data["rule"] == "cholesterol > 240"
        assert data["delta"] == -0.25
        assert data["reason"] == "Logic validation failed: Rule violation"
        
        # Test deserialization
        restored = ColumnInsight.from_dict(data)
        assert restored.column == insight.column
        assert restored.rule == insight.rule
        assert restored.delta == insight.delta
        assert restored.reason == insight.reason
    
    def test_block_row_diagnostic_creation(self):
        """Test BlockRowDiagnostic creation and serialization."""
        columns = [
            ColumnInsight(column="feature_1", rule="value > 0.5", delta=-0.1),
            ColumnInsight(column="feature_2", rule="value < -0.5", delta=-0.15)
        ]
        
        diagnostic = BlockRowDiagnostic(
            run_id="test_run_123",
            block_index=0,
            row_id=42,
            action="DOWN_WEIGHTED",
            vq=0.8,
            vb=0.6,
            vl=0.4,
            weight_delta=-0.25,
            columns=columns,
            reason="High entropy detected"
        )
        
        assert diagnostic.run_id == "test_run_123"
        assert diagnostic.block_index == 0
        assert diagnostic.row_id == 42
        assert diagnostic.action == "DOWN_WEIGHTED"
        assert diagnostic.vq == 0.8
        assert diagnostic.vb == 0.6
        assert diagnostic.vl == 0.4
        assert diagnostic.weight_delta == -0.25
        assert len(diagnostic.columns) == 2
        assert diagnostic.reason == "High entropy detected"
        
        # Test serialization
        data = diagnostic.to_dict()
        assert data["run_id"] == "test_run_123"
        assert data["block_index"] == 0
        assert data["row_id"] == 42
        assert data["action"] == "DOWN_WEIGHTED"
        assert data["vq"] == 0.8
        assert data["vb"] == 0.6
        assert data["vl"] == 0.4
        assert data["weight_delta"] == -0.25
        assert len(data["columns"]) == 2
        assert data["reason"] == "High entropy detected"
        
        # Test deserialization
        restored = BlockRowDiagnostic.from_dict(data)
        assert restored.run_id == diagnostic.run_id
        assert restored.block_index == diagnostic.block_index
        assert restored.row_id == diagnostic.row_id
        assert restored.action == diagnostic.action
        assert restored.vq == diagnostic.vq
        assert restored.vb == diagnostic.vb
        assert restored.vl == diagnostic.vl
        assert restored.weight_delta == diagnostic.weight_delta
        assert len(restored.columns) == len(diagnostic.columns)
        assert restored.reason == diagnostic.reason
    
    def test_block_diagnostic_summary_creation(self):
        """Test BlockDiagnosticSummary creation and serialization."""
        summary = BlockDiagnosticSummary(
            run_id="test_run_123",
            block_index=0,
            affected_rows=10,
            avg_vq=0.75,
            avg_vb=0.65,
            avg_vl=0.55,
            actions_count={"DOWN_WEIGHTED": 3, "RETAINED": 6, "FLAGGED": 1}
        )
        
        assert summary.run_id == "test_run_123"
        assert summary.block_index == 0
        assert summary.affected_rows == 10
        assert summary.avg_vq == 0.75
        assert summary.avg_vb == 0.65
        assert summary.avg_vl == 0.55
        assert summary.actions_count["DOWN_WEIGHTED"] == 3
        assert summary.actions_count["RETAINED"] == 6
        assert summary.actions_count["FLAGGED"] == 1
        
        # Test serialization
        data = summary.to_dict()
        assert data["run_id"] == "test_run_123"
        assert data["block_index"] == 0
        assert data["affected_rows"] == 10
        assert data["avg_vq"] == 0.75
        assert data["avg_vb"] == 0.65
        assert data["avg_vl"] == 0.55
        assert data["actions_count"]["DOWN_WEIGHTED"] == 3
        assert data["actions_count"]["RETAINED"] == 6
        assert data["actions_count"]["FLAGGED"] == 1
        
        # Test deserialization
        restored = BlockDiagnosticSummary.from_dict(data)
        assert restored.run_id == summary.run_id
        assert restored.block_index == summary.block_index
        assert restored.affected_rows == summary.affected_rows
        assert restored.avg_vq == summary.avg_vq
        assert restored.avg_vb == summary.avg_vb
        assert restored.avg_vl == summary.avg_vl
        assert restored.actions_count == summary.actions_count


class TestDiagnosticsService:
    """Test DiagnosticsService functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DiagnosticsConfig(
            enabled=True,
            persist=False,
            max_rows_per_block=1000
        )
        self.service = DiagnosticsService(self.config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_service_initialization_disabled(self):
        """Test service initialization with diagnostics disabled."""
        config = DiagnosticsConfig(enabled=False)
        service = DiagnosticsService(config)
        
        assert not service.config.enabled
        assert service.config.persist is False
        assert service.config.max_rows_per_block == 10000
    
    def test_service_initialization_enabled(self):
        """Test service initialization with diagnostics enabled."""
        config = DiagnosticsConfig(enabled=True, persist=True)
        service = DiagnosticsService(config)
        
        assert service.config.enabled
        assert service.config.persist is True
        assert service.config.max_rows_per_block == 10000
    
    def test_emit_row_disabled(self):
        """Test that emit_row does nothing when disabled."""
        config = DiagnosticsConfig(enabled=False)
        service = DiagnosticsService(config)
        
        # Should not raise any errors
        service.emit_row("test_run", 0, {
            'row_id': 1,
            'action': 'RETAINED',
            'vq': 0.8,
            'vb': 0.7,
            'vl': 0.6
        })
        
        # No diagnostics should be stored
        assert len(service._diagnostics) == 0
    
    def test_emit_row_enabled(self):
        """Test that emit_row stores diagnostics when enabled."""
        payload = {
            'row_id': 42,
            'action': 'DOWN_WEIGHTED',
            'vq': 0.8,
            'vb': 0.6,
            'vl': 0.4,
            'weight_delta': -0.25,
            'reason': 'High entropy detected'
        }
        
        self.service.emit_row("test_run_123", 0, payload)
        
        # Check that diagnostic was stored
        assert "test_run_123" in self.service._diagnostics
        assert len(self.service._diagnostics["test_run_123"]) == 1
        
        diagnostic = self.service._diagnostics["test_run_123"][0]
        assert diagnostic.run_id == "test_run_123"
        assert diagnostic.block_index == 0
        assert diagnostic.row_id == 42
        assert diagnostic.action == "DOWN_WEIGHTED"
        assert diagnostic.vq == 0.8
        assert diagnostic.vb == 0.6
        assert diagnostic.vl == 0.4
        assert diagnostic.weight_delta == -0.25
        assert diagnostic.reason == "High entropy detected"
    
    def test_max_rows_per_block_limit(self):
        """Test that max rows per block limit is enforced."""
        config = DiagnosticsConfig(enabled=True, max_rows_per_block=5)
        service = DiagnosticsService(config)
        
        # Emit more rows than the limit
        for i in range(10):
            service.emit_row("test_run", 0, {
                'row_id': i,
                'action': 'RETAINED'
            })
        
        # Should have limited the number of stored diagnostics
        assert len(service._diagnostics["test_run"]) <= 5
    
    def test_summarize_block_empty(self):
        """Test summarizing an empty block."""
        summary = self.service.summarize_block("test_run", 0)
        
        assert summary.run_id == "test_run"
        assert summary.block_index == 0
        assert summary.affected_rows == 0
        assert summary.avg_vq is None
        assert summary.avg_vb is None
        assert summary.avg_vl is None
        assert summary.actions_count["DOWN_WEIGHTED"] == 0
        assert summary.actions_count["RETAINED"] == 0
        assert summary.actions_count["FLAGGED"] == 0
    
    def test_summarize_block_with_data(self):
        """Test summarizing a block with diagnostic data."""
        # Emit some diagnostics
        for i in range(5):
            self.service.emit_row("test_run", 0, {
                'row_id': i,
                'action': 'DOWN_WEIGHTED' if i < 2 else 'RETAINED',
                'vq': 0.8 + i * 0.1,
                'vb': 0.6 + i * 0.1,
                'vl': 0.4 + i * 0.1
            })
        
        summary = self.service.summarize_block("test_run", 0)
        
        assert summary.run_id == "test_run"
        assert summary.block_index == 0
        assert summary.affected_rows == 5
        assert summary.avg_vq is not None
        assert summary.avg_vb is not None
        assert summary.avg_vl is not None
        assert summary.actions_count["DOWN_WEIGHTED"] == 2
        assert summary.actions_count["RETAINED"] == 3
        assert summary.actions_count["FLAGGED"] == 0
    
    def test_get_block_diagnostics(self):
        """Test retrieving block diagnostics."""
        # Emit diagnostics for multiple blocks
        self.service.emit_row("test_run", 0, {'row_id': 1, 'action': 'RETAINED'})
        self.service.emit_row("test_run", 1, {'row_id': 2, 'action': 'DOWN_WEIGHTED'})
        self.service.emit_row("test_run", 1, {'row_id': 3, 'action': 'RETAINED'})
        
        # Get diagnostics for block 1
        diagnostics = self.service.get_block_diagnostics("test_run", 1)
        assert len(diagnostics) == 2
        assert all(d.block_index == 1 for d in diagnostics)
    
    def test_get_block_summaries(self):
        """Test retrieving block summaries."""
        # Emit diagnostics and create summaries
        self.service.emit_row("test_run", 0, {'row_id': 1, 'action': 'RETAINED'})
        self.service.emit_row("test_run", 1, {'row_id': 2, 'action': 'DOWN_WEIGHTED'})
        
        self.service.summarize_block("test_run", 0)
        self.service.summarize_block("test_run", 1)
        
        summaries = self.service.get_block_summaries("test_run")
        assert len(summaries) == 2
        assert all(s.run_id == "test_run" for s in summaries)
    
    def test_clear_run(self):
        """Test clearing diagnostics for a run."""
        # Emit some diagnostics
        self.service.emit_row("test_run", 0, {'row_id': 1, 'action': 'RETAINED'})
        self.service.summarize_block("test_run", 0)
        
        # Verify data exists
        assert "test_run" in self.service._diagnostics
        assert "test_run" in self.service._summaries
        
        # Clear the run
        self.service.clear_run("test_run")
        
        # Verify data is cleared
        assert "test_run" not in self.service._diagnostics
        assert "test_run" not in self.service._summaries
    
    def test_get_stats_disabled(self):
        """Test getting stats when diagnostics are disabled."""
        config = DiagnosticsConfig(enabled=False)
        service = DiagnosticsService(config)
        
        stats = service.get_stats()
        assert stats["enabled"] is False
    
    def test_get_stats_enabled(self):
        """Test getting stats when diagnostics are enabled."""
        # Emit some diagnostics
        self.service.emit_row("test_run", 0, {'row_id': 1, 'action': 'RETAINED'})
        self.service.summarize_block("test_run", 0)
        
        stats = self.service.get_stats()
        assert stats["enabled"] is True
        assert stats["total_runs"] == 1
        assert stats["total_diagnostics"] == 1
        assert stats["total_summaries"] == 1


class TestDiagnosticsHooks:
    """Test DiagnosticsHooks functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = DiagnosticsConfig(enabled=True)
        self.service = DiagnosticsService(self.config)
        
        # Mock the global service
        with patch('diagnostics.hooks.get_diagnostics_service', return_value=self.service):
            from diagnostics.hooks import DiagnosticsHooks
            self.hooks = DiagnosticsHooks()
    
    def test_emit_block_start(self):
        """Test emitting block start diagnostics."""
        # Should not raise any errors
        self.hooks.emit_block_start("test_run", 0, 100)
    
    def test_emit_row_validation(self):
        """Test emitting row validation diagnostics."""
        self.hooks.emit_row_validation(
            run_id="test_run",
            block_index=0,
            row_id=42,
            vq=0.8,
            vb=0.6,
            vl=0.4,
            action="DOWN_WEIGHTED",
            weight_delta=-0.25,
            reason="High entropy detected"
        )
        
        # Check that diagnostic was emitted
        diagnostics = self.service.get_block_diagnostics("test_run", 0)
        assert len(diagnostics) == 1
        
        diagnostic = diagnostics[0]
        assert diagnostic.run_id == "test_run"
        assert diagnostic.block_index == 0
        assert diagnostic.row_id == 42
        assert diagnostic.vq == 0.8
        assert diagnostic.vb == 0.6
        assert diagnostic.vl == 0.4
        assert diagnostic.action == "DOWN_WEIGHTED"
        assert diagnostic.weight_delta == -0.25
        assert diagnostic.reason == "High entropy detected"
    
    def test_emit_block_end(self):
        """Test emitting block end diagnostics."""
        # Emit some diagnostics first
        self.hooks.emit_row_validation("test_run", 0, 1, action="RETAINED")
        self.hooks.emit_row_validation("test_run", 0, 2, action="DOWN_WEIGHTED")
        
        # Emit block end
        summary = self.hooks.emit_block_end("test_run", 0)
        
        assert summary is not None
        assert summary["run_id"] == "test_run"
        assert summary["block_index"] == 0
        assert summary["affected_rows"] == 2
        assert summary["actions_count"]["RETAINED"] == 1
        assert summary["actions_count"]["DOWN_WEIGHTED"] == 1
    
    def test_emit_logic_validation(self):
        """Test emitting logic validation diagnostics."""
        feature_names = ["feature_1", "feature_2", "feature_3"]
        feature_values = np.array([0.8, -0.5, 1.2])
        logic_rules = [
            {"feature_index": 0, "description": "feature_1 > 0.5", "reason": "Rule violation"},
            {"feature_index": 1, "description": "feature_2 < 0", "reason": "Rule violation"}
        ]
        validation_results = [True, False]  # Second rule fails
        
        insights = self.hooks.emit_logic_validation(
            run_id="test_run",
            block_index=0,
            row_id=42,
            feature_names=feature_names,
            feature_values=feature_values,
            logic_rules=logic_rules,
            validation_results=validation_results,
            weight_delta=-0.25
        )
        
        assert len(insights) == 1  # Only one rule failed
        insight = insights[0]
        assert insight.column == "feature_2"
        assert "feature_2 < 0" in insight.rule
        assert insight.delta == -0.25
        assert "Logic validation failed" in insight.reason
    
    def test_emit_entropy_validation(self):
        """Test emitting entropy validation diagnostics."""
        self.hooks.emit_entropy_validation(
            run_id="test_run",
            block_index=0,
            row_id=42,
            entropy_score=2.5,
            threshold=2.0,
            weight_delta=-0.25
        )
        
        diagnostics = self.service.get_block_diagnostics("test_run", 0)
        assert len(diagnostics) == 1
        
        diagnostic = diagnostics[0]
        assert diagnostic.vq == 2.5
        assert diagnostic.action == "DOWN_WEIGHTED"
        assert diagnostic.weight_delta == -0.25
        assert "exceeds" in diagnostic.reason
    
    def test_emit_hash_validation(self):
        """Test emitting hash validation diagnostics."""
        self.hooks.emit_hash_validation(
            run_id="test_run",
            block_index=0,
            row_id=42,
            hash_score=1.5,
            threshold=1.0,
            weight_delta=-0.25
        )
        
        diagnostics = self.service.get_block_diagnostics("test_run", 0)
        assert len(diagnostics) == 1
        
        diagnostic = diagnostics[0]
        assert diagnostic.vb == 1.5
        assert diagnostic.action == "DOWN_WEIGHTED"
        assert diagnostic.weight_delta == -0.25
        assert "exceeds" in diagnostic.reason
    
    def test_emit_flagged_row(self):
        """Test emitting flagged row diagnostics."""
        self.hooks.emit_flagged_row(
            run_id="test_run",
            block_index=0,
            row_id=42,
            reason="Multiple validation failures",
            vq=0.3,
            vb=0.2,
            vl=0.1
        )
        
        diagnostics = self.service.get_block_diagnostics("test_run", 0)
        assert len(diagnostics) == 1
        
        diagnostic = diagnostics[0]
        assert diagnostic.action == "FLAGGED"
        assert diagnostic.reason == "Multiple validation failures"
        assert diagnostic.vq == 0.3
        assert diagnostic.vb == 0.2
        assert diagnostic.vl == 0.1


class TestDiagnosticsAPI:
    """Test DiagnosticsAPI functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = DiagnosticsConfig(enabled=True)
        self.service = DiagnosticsService(self.config)
        
        # Mock the global service
        with patch('diagnostics.api.get_diagnostics_service', return_value=self.service):
            self.api = DiagnosticsAPI()
    
    def test_get_block_summaries(self):
        """Test getting block summaries via API."""
        # Create some summaries
        self.service.emit_row("test_run", 0, {'row_id': 1, 'action': 'RETAINED'})
        self.service.summarize_block("test_run", 0)
        
        summaries = self.api.get_block_summaries("test_run")
        assert len(summaries) == 1
        assert summaries[0]["run_id"] == "test_run"
        assert summaries[0]["block_index"] == 0
    
    def test_get_block_diagnostics(self):
        """Test getting block diagnostics via API."""
        # Create some diagnostics
        self.service.emit_row("test_run", 0, {'row_id': 1, 'action': 'RETAINED'})
        self.service.emit_row("test_run", 0, {'row_id': 2, 'action': 'DOWN_WEIGHTED'})
        
        diagnostics = self.api.get_block_diagnostics("test_run", 0)
        assert len(diagnostics) == 2
        assert all(d["run_id"] == "test_run" for d in diagnostics)
        assert all(d["block_index"] == 0 for d in diagnostics)
    
    def test_get_stats(self):
        """Test getting stats via API."""
        stats = self.api.get_stats()
        assert stats["enabled"] is True
    
    def test_clear_run(self):
        """Test clearing run via API."""
        # Create some data
        self.service.emit_row("test_run", 0, {'row_id': 1, 'action': 'RETAINED'})
        
        result = self.api.clear_run("test_run")
        assert result["success"] is True
        assert result["run_id"] == "test_run"
        
        # Verify data is cleared
        diagnostics = self.api.get_block_diagnostics("test_run", 0)
        assert len(diagnostics) == 0


class TestFeatureFlagging:
    """Test feature flagging behavior."""
    
    def test_feature_disabled_by_default(self):
        """Test that diagnostics are disabled by default."""
        service = DiagnosticsService()  # No config provided
        assert not service.config.enabled
    
    def test_no_side_effects_when_disabled(self):
        """Test that no side effects occur when diagnostics are disabled."""
        config = DiagnosticsConfig(enabled=False)
        service = DiagnosticsService(config)
        
        # These operations should not affect the system
        service.emit_row("test_run", 0, {'row_id': 1, 'action': 'RETAINED'})
        summary = service.summarize_block("test_run", 0)
        diagnostics = service.get_block_diagnostics("test_run", 0)
        
        # Verify no data was stored
        assert len(service._diagnostics) == 0
        assert len(service._summaries) == 0
        assert summary.affected_rows == 0
        assert len(diagnostics) == 0
    
    def test_hooks_no_op_when_disabled(self):
        """Test that hooks do nothing when diagnostics are disabled."""
        config = DiagnosticsConfig(enabled=False)
        service = DiagnosticsService(config)
        
        with patch('diagnostics.hooks.get_diagnostics_service', return_value=service):
            from diagnostics.hooks import DiagnosticsHooks
            hooks = DiagnosticsHooks()
            
            # These should not raise errors or store data
            hooks.emit_block_start("test_run", 0, 100)
            hooks.emit_row_validation("test_run", 0, 1, action="RETAINED")
            hooks.emit_block_end("test_run", 0)
            
            # Verify no data was stored
            assert len(service._diagnostics) == 0
            assert len(service._summaries) == 0


if __name__ == "__main__":
    pytest.main([__file__]) 