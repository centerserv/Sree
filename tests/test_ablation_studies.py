#!/usr/bin/env python3
"""
Test suite for Ablation Studies framework.
Tests the comprehensive ablation study functionality.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ablation_studies import AblationStudy
from config import setup_logging


class TestAblationStudy:
    """Test cases for AblationStudy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_logging(level="WARNING")  # Reduce log noise during tests
        self.ablation = AblationStudy(self.logger)
        
        # Create mock data
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_ablation_study_initialization(self):
        """Test AblationStudy initialization."""
        assert self.ablation.logger is not None
        assert self.ablation.results == {}
        assert self.ablation.results_dir.exists()
    
    def test_create_layer_combinations(self):
        """Test layer combination creation."""
        combinations = self.ablation.create_layer_combinations()
        
        # Check structure
        assert "individual" in combinations
        assert "pairs" in combinations
        assert "triplets" in combinations
        assert "full_ppp" in combinations
        
        # Check individual layers
        individual = combinations["individual"]
        assert len(individual) == 4
        assert ["pattern"] in individual
        assert ["presence"] in individual
        assert ["permanence"] in individual
        assert ["logic"] in individual
        
        # Check pairs
        pairs = combinations["pairs"]
        assert len(pairs) == 6  # 4C2 = 6 combinations
        
        # Check full PPP
        full_ppp = combinations["full_ppp"]
        assert len(full_ppp) == 1
        assert full_ppp[0] == ["pattern", "presence", "permanence", "logic"]
    
    def test_get_validators_for_combination(self):
        """Test validator creation for layer combinations."""
        # Test individual layers
        pattern_validators = self.ablation.get_validators_for_combination(["pattern"])
        assert len(pattern_validators) == 1
        assert pattern_validators[0].__class__.__name__ == "PatternValidator"
        
        # Test multiple layers
        multi_validators = self.ablation.get_validators_for_combination(["pattern", "presence"])
        assert len(multi_validators) == 2
        assert multi_validators[0].__class__.__name__ == "PatternValidator"
        assert multi_validators[1].__class__.__name__ == "PresenceValidator"
        
        # Test full PPP
        full_validators = self.ablation.get_validators_for_combination(
            ["pattern", "presence", "permanence", "logic"]
        )
        assert len(full_validators) == 4
    
    @patch('ablation_studies.TrustUpdateLoop')
    def test_run_single_ablation_test(self, mock_trust_loop):
        """Test single ablation test execution."""
        # Mock trust loop results
        mock_results = {
            "final_accuracy": 0.85,
            "final_trust": 0.92,
            "convergence_achieved": True,
            "iterations": [
                {
                    "pattern_trust": 0.88,
                    "presence_trust": 0.90,
                    "permanence_trust": 0.87,
                    "logic_trust": 0.89
                }
            ]
        }
        mock_trust_loop.return_value.run_ppp_loop.return_value = mock_results
        
        # Run test
        result = self.ablation.run_single_ablation_test(
            ["pattern", "presence"], 
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        
        # Check results
        assert result["combination"] == "pattern+presence"
        assert result["layers"] == ["pattern", "presence"]
        assert result["n_layers"] == 2
        assert result["final_accuracy"] == 0.85
        assert result["final_trust"] == 0.92
        assert result["convergence_achieved"] is True
        assert result["iterations_completed"] == 1
        assert result["pattern_trust"] == 0.88
        assert result["presence_trust"] == 0.90
    
    @patch('ablation_studies.DataLoader')
    def test_run_comprehensive_ablation(self, mock_data_loader):
        """Test comprehensive ablation study execution."""
        # Mock data loader
        mock_loader = Mock()
        mock_loader.load_mnist.return_value = (self.X_train, self.y_train)
        mock_loader.preprocess_data.return_value = (
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        mock_data_loader.return_value = mock_loader
        
        # Mock trust loop results
        with patch('ablation_studies.TrustUpdateLoop') as mock_trust_loop:
            mock_results = {
                "final_accuracy": 0.88,
                "final_trust": 0.94,
                "convergence_achieved": True,
                "iterations": [
                    {
                        "pattern_trust": 0.90,
                        "presence_trust": 0.92,
                        "permanence_trust": 0.89,
                        "logic_trust": 0.91
                    }
                ]
            }
            mock_trust_loop.return_value.run_ppp_loop.return_value = mock_results
            
            # Run comprehensive ablation
            results = self.ablation.run_comprehensive_ablation("mnist")
            
            # Check results structure
            assert "dataset" in results
            assert "results" in results
            assert "analysis" in results
            assert results["dataset"] == "mnist"
            assert len(results["results"]) > 0
    
    def test_analyze_ablation_results(self):
        """Test ablation results analysis."""
        # Create mock results
        mock_results = [
            {
                "combination": "pattern",
                "category": "individual",
                "final_accuracy": 0.75,
                "final_trust": 0.80,
                "convergence_achieved": True,
                "pattern_trust": 0.80,
                "presence_trust": 0.0,
                "permanence_trust": 0.0,
                "logic_trust": 0.0
            },
            {
                "combination": "presence",
                "category": "individual",
                "final_accuracy": 0.70,
                "final_trust": 0.75,
                "convergence_achieved": True,
                "pattern_trust": 0.0,
                "presence_trust": 0.75,
                "permanence_trust": 0.0,
                "logic_trust": 0.0
            },
            {
                "combination": "pattern+presence+permanence+logic",
                "category": "full_ppp",
                "final_accuracy": 0.85,
                "final_trust": 0.90,
                "convergence_achieved": True,
                "pattern_trust": 0.88,
                "presence_trust": 0.89,
                "permanence_trust": 0.87,
                "logic_trust": 0.86
            }
        ]
        
        # Analyze results
        analysis = self.ablation.analyze_ablation_results(mock_results)
        
        # Check analysis structure
        assert "full_ppp_performance" in analysis
        assert "best_individual" in analysis
        assert "synergy_metrics" in analysis
        assert "layer_contributions" in analysis
        
        # Check synergy calculations
        synergy = analysis["synergy_metrics"]
        assert abs(synergy["accuracy_synergy"] - 0.10) < 1e-10  # 0.85 - 0.75
        assert abs(synergy["trust_synergy"] - 0.10) < 1e-10     # 0.90 - 0.80
        assert bool(synergy["synergy_achieved"]) is True
    
    def test_save_ablation_results(self):
        """Test ablation results saving."""
        # Create mock results and analysis
        mock_results = [
            {
                "combination": "pattern",
                "final_accuracy": 0.75,
                "final_trust": 0.80
            }
        ]
        
        mock_analysis = {
            "synergy_metrics": {
                "synergy_achieved": True
            }
        }
        
        # Save results
        self.ablation.save_ablation_results("test", mock_results, mock_analysis)
        
        # Check files were created
        json_file = self.ablation.results_dir / "ablation_results_test.json"
        csv_file = self.ablation.results_dir / "ablation_results_test.csv"
        
        assert json_file.exists()
        assert csv_file.exists()
        
        # Clean up
        json_file.unlink(missing_ok=True)
        csv_file.unlink(missing_ok=True)
    
    def test_generate_ablation_summary(self):
        """Test ablation summary generation."""
        # Create mock result file
        mock_data = {
            "analysis": {
                "synergy_metrics": {
                    "synergy_achieved": True,
                    "accuracy_synergy": 0.10,
                    "trust_synergy": 0.15
                }
            }
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = str(mock_data)
            
            # Mock json.load
            with patch('json.load') as mock_json_load:
                mock_json_load.return_value = mock_data
                
                # Mock glob
                with patch('pathlib.Path.glob') as mock_glob:
                    mock_glob.return_value = [Path("ablation_results_test.json")]
                    
                    # Generate summary
                    summary = self.ablation.generate_ablation_summary()
                    
                    # Check summary structure
                    assert summary["study_completed"] is True
                    assert summary["overall_synergy_achieved"] is True
                    assert len(summary["datasets_tested"]) > 0
                    assert len(summary["key_findings"]) > 0
                    assert len(summary["recommendations"]) > 0


class TestAblationStudyIntegration:
    """Integration tests for ablation studies."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.logger = setup_logging(level="WARNING")
        self.ablation = AblationStudy(self.logger)
    
    @patch('ablation_studies.DataLoader')
    def test_full_ablation_workflow(self, mock_data_loader):
        """Test complete ablation workflow."""
        # Mock data loader
        mock_loader = Mock()
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 10)
        y_test = np.random.randint(0, 2, 20)
        
        mock_loader.load_mnist.return_value = (X_train, y_train)
        mock_loader.preprocess_data.return_value = (X_train, X_test, y_train, y_test)
        mock_data_loader.return_value = mock_loader
        
        # Mock trust loop with realistic results
        with patch('ablation_studies.TrustUpdateLoop') as mock_trust_loop:
            # Create realistic mock results
            def create_mock_results(X_train, y_train, X_test, y_test):
                base_accuracy = 0.70
                base_trust = 0.75
                
                # Add synergy for combinations (simulate based on data size)
                if X_train.shape[0] > 50:  # More data = better synergy
                    base_accuracy += 0.05
                    base_trust += 0.05
                
                return {
                    "final_accuracy": base_accuracy,
                    "final_trust": base_trust,
                    "convergence_achieved": True,
                    "iterations": [
                        {
                            "pattern_trust": 0.80,
                            "presence_trust": 0.82,
                            "permanence_trust": 0.78,
                            "logic_trust": 0.81
                        }
                    ]
                }
            
            mock_trust_loop.return_value.run_ppp_loop.side_effect = create_mock_results
            
            # Run comprehensive ablation
            results = self.ablation.run_comprehensive_ablation("mnist")
            
            # Verify results
            assert results["dataset"] == "mnist"
            assert len(results["results"]) > 0
            assert "analysis" in results
            
            # Check that synergy is calculated
            analysis = results["analysis"]
            assert "synergy_metrics" in analysis
            assert "synergy_achieved" in analysis["synergy_metrics"]


def test_ablation_study_main():
    """Test the main ablation study execution."""
    with patch('ablation_studies.AblationStudy') as mock_ablation_class:
        mock_ablation = Mock()
        mock_ablation_class.return_value = mock_ablation
        
        # Mock comprehensive ablation results
        mock_results = {
            "analysis": {
                "synergy_metrics": {
                    "synergy_achieved": True
                }
            }
        }
        mock_ablation.run_comprehensive_ablation.return_value = mock_results
        mock_ablation.generate_ablation_summary.return_value = {
            "overall_synergy_achieved": True,
            "datasets_tested": [{"dataset": "test", "synergy_achieved": True}],
            "key_findings": ["Test finding"],
            "recommendations": ["Test recommendation"]
        }
        
        # Import and test main function
        from ablation_studies import main
        result = main()
        
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 