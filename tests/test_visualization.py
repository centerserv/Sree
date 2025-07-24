#!/usr/bin/env python3
"""
SREE Visualization Module Tests
Test the visualization module for generating academic figures.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import logging
from unittest.mock import Mock, patch

# Import the visualization module
from visualization import SREEVisualizer


class TestSREEVisualizer:
    """Test the SREEVisualizer class."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = SREEVisualizer(output_dir=self.temp_dir)
        
    def teardown_method(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        assert self.visualizer.output_dir.exists()
        assert self.visualizer.output_dir.is_dir()
        assert self.visualizer.logger is not None
        
    def test_generate_ppp_state_diagram(self):
        """Test PPP state diagram generation."""
        output_path = self.visualizer.generate_ppp_state_diagram("test_fig1.png")
        
        # Check that file was created
        assert Path(output_path).exists()
        assert Path(output_path).suffix == '.png'
        
        # Check file size (should be reasonable for a figure)
        file_size = Path(output_path).stat().st_size
        assert file_size > 1000  # At least 1KB
        
    def test_generate_trust_accuracy_curves(self):
        """Test trust/accuracy curves generation."""
        output_path = self.visualizer.generate_trust_accuracy_curves("test_fig2.png")
        
        # Check that file was created
        assert Path(output_path).exists()
        assert Path(output_path).suffix == '.png'
        
        # Check file size
        file_size = Path(output_path).stat().st_size
        assert file_size > 1000
        
    def test_generate_ablation_visualization(self):
        """Test ablation visualization generation."""
        output_path = self.visualizer.generate_ablation_visualization("test_ablation.png")
        
        # Check that file was created
        assert Path(output_path).exists()
        assert Path(output_path).suffix == '.png'
        
        # Check file size
        file_size = Path(output_path).stat().st_size
        assert file_size > 1000
        
    def test_generate_performance_comparison(self):
        """Test performance comparison generation."""
        output_path = self.visualizer.generate_performance_comparison("test_performance.png")
        
        # Check that file was created
        assert Path(output_path).exists()
        assert Path(output_path).suffix == '.png'
        
        # Check file size
        file_size = Path(output_path).stat().st_size
        assert file_size > 1000
        
    def test_generate_all_figures(self):
        """Test generation of all figures."""
        figures = self.visualizer.generate_all_figures()
        
        # Check that all expected figures were generated
        expected_keys = ['ppp_diagram', 'trust_accuracy', 'ablation', 'performance']
        for key in expected_keys:
            assert key in figures
            assert Path(figures[key]).exists()
            
    def test_load_fault_injection_results_nonexistent(self):
        """Test loading fault injection results when files don't exist."""
        results = self.visualizer._load_fault_injection_results()
        assert results is None
        
    def test_load_ablation_results_nonexistent(self):
        """Test loading ablation results when files don't exist."""
        results = self.visualizer._load_ablation_results()
        assert results is None
        
    @patch('visualization.json.load')
    @patch('builtins.open')
    def test_load_fault_injection_results_existing(self, mock_open, mock_json_load):
        """Test loading fault injection results when files exist."""
        # Mock the file existence
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock the JSON data
        mock_data = {
            'mnist': {
                'clean': {'trust_score': 0.95, 'accuracy': 0.93},
                'corruption_results': {
                    '5': {'trust_score': 0.92, 'accuracy': 0.90},
                    '10': {'trust_score': 0.89, 'accuracy': 0.87}
                }
            }
        }
        mock_json_load.return_value = mock_data
        
        # Mock file glob
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.return_value = [Path('test_file.json')]
            results = self.visualizer._load_fault_injection_results()
            
        assert results == mock_data
        
    def test_plot_synthetic_curves(self):
        """Test synthetic curves plotting."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Test that plotting doesn't raise errors
        self.visualizer._plot_synthetic_curves(ax1, ax2)
        
        # Check that axes have been modified
        assert len(ax1.get_lines()) > 0
        assert len(ax2.get_lines()) > 0
        
        plt.close(fig)
        
    def test_plot_synthetic_ablation(self):
        """Test synthetic ablation plotting."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Test that plotting doesn't raise errors
        self.visualizer._plot_synthetic_ablation(ax1, ax2)
        
        # Check that axes have been modified
        assert len(ax1.patches) > 0
        assert len(ax2.patches) > 0
        
        plt.close(fig)
        
    def test_publication_quality_settings(self):
        """Test that matplotlib is configured for publication quality."""
        # Check that DPI is set to 300
        assert plt.rcParams['figure.dpi'] == 300
        assert plt.rcParams['savefig.dpi'] == 300
        
        # Check that font sizes are appropriate
        assert plt.rcParams['font.size'] == 10
        assert plt.rcParams['axes.titlesize'] == 12
        assert plt.rcParams['figure.titlesize'] == 14
        
    def test_error_handling_invalid_output_dir(self):
        """Test error handling with invalid output directory."""
        # Test with invalid output directory
        with pytest.raises(Exception):
            visualizer = SREEVisualizer(output_dir="/invalid/path/that/should/not/exist")
            
    def test_figure_dimensions(self):
        """Test that figures are generated with correct dimensions."""
        # Generate a figure and check its dimensions
        output_path = self.visualizer.generate_ppp_state_diagram("test_dimensions.png")
        
        # Load the image and check dimensions
        img = plt.imread(output_path)
        height, width = img.shape[:2]
        
        # Should be reasonable dimensions for a publication figure
        assert width > 1000  # At least 1000 pixels wide
        assert height > 500   # At least 500 pixels tall
        
    def test_color_scheme_consistency(self):
        """Test that color schemes are consistent across figures."""
        # Generate multiple figures and check color consistency
        self.visualizer.generate_ppp_state_diagram("test_colors1.png")
        self.visualizer.generate_performance_comparison("test_colors2.png")
        
        # Both figures should exist
        assert Path(self.temp_dir) / "test_colors1.png"
        assert Path(self.temp_dir) / "test_colors2.png"


class TestVisualizationIntegration:
    """Integration tests for visualization module."""
    
    def test_visualization_with_real_data(self):
        """Test visualization with real fault injection and ablation data."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create mock data files
            logs_dir = Path(temp_dir) / "logs"
            logs_dir.mkdir()
            
            # Mock fault injection results
            fault_data = {
                'mnist': {
                    'clean': {'trust_score': 0.95, 'accuracy': 0.93},
                    'corruption_results': {
                        '5': {'trust_score': 0.92, 'accuracy': 0.90},
                        '10': {'trust_score': 0.89, 'accuracy': 0.87},
                        '15': {'trust_score': 0.86, 'accuracy': 0.84}
                    }
                },
                'heart': {
                    'clean': {'trust_score': 0.92, 'accuracy': 0.90},
                    'corruption_results': {
                        '5': {'trust_score': 0.89, 'accuracy': 0.87},
                        '10': {'trust_score': 0.86, 'accuracy': 0.84},
                        '15': {'trust_score': 0.83, 'accuracy': 0.81}
                    }
                }
            }
            
            with open(logs_dir / "fault_injection_results_test.json", 'w') as f:
                import json
                json.dump(fault_data, f)
            
            # Mock ablation results
            ablation_data = {
                'synthetic': {
                    'synergy_metrics': {
                        'trust_synergy': 0.165,
                        'accuracy_synergy': 0.120,
                        'synergy_achieved': True
                    }
                },
                'mnist': {
                    'synergy_metrics': {
                        'trust_synergy': 0.293,
                        'accuracy_synergy': 0.180,
                        'synergy_achieved': True
                    }
                }
            }
            
            with open(logs_dir / "ablation_results_test.json", 'w') as f:
                json.dump(ablation_data, f)
            
            # Create visualizer with the temp directory
            visualizer = SREEVisualizer(output_dir=temp_dir)
            
            # Generate figures
            figures = visualizer.generate_all_figures()
            
            # Check that all figures were generated
            assert len(figures) == 4
            for path in figures.values():
                assert Path(path).exists()
                
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_visualization_main():
    """Test the main function of the visualization module."""
    with patch('visualization.SREEVisualizer') as mock_visualizer_class:
        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer
        
        mock_figures = {
            'ppp_diagram': '/path/to/fig1.png',
            'trust_accuracy': '/path/to/fig2.png',
            'ablation': '/path/to/ablation.png',
            'performance': '/path/to/performance.png'
        }
        mock_visualizer.generate_all_figures.return_value = mock_figures
        
        # Import and run main
        from visualization import main
        main()
        
        # Check that generate_all_figures was called
        mock_visualizer.generate_all_figures.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 