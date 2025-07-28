#!/usr/bin/env python3
"""
Test Dashboard Display
Script to test dashboard display with correct values.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_dashboard_display():
    """Test dashboard display with mock results."""
    logger = logging.getLogger(__name__)
    
    # Mock results that should be displayed correctly
    mock_results = {
        'accuracy': 0.964912,
        'trust_score': 0.992979,
        'entropy': 1.500000,
        'block_count': 11,
        'accuracy_ok': True,
        'trust_ok': True,
        'entropy_ok': True,
        'ppp_results': {
            'iterations': [
                {'iteration': 1, 'accuracy': 0.9649, 'updated_trust': 0.4210},
                {'iteration': 2, 'accuracy': 0.9649, 'updated_trust': 0.8214},
                {'iteration': 3, 'accuracy': 0.9649, 'updated_trust': 0.9415},
                {'iteration': 4, 'accuracy': 0.9649, 'updated_trust': 0.9775},
                {'iteration': 5, 'accuracy': 0.9649, 'updated_trust': 0.9883},
                {'iteration': 6, 'accuracy': 0.9649, 'updated_trust': 0.9916},
                {'iteration': 7, 'accuracy': 0.9649, 'updated_trust': 0.9926},
                {'iteration': 8, 'accuracy': 0.9649, 'updated_trust': 0.9929},
                {'iteration': 9, 'accuracy': 0.9649, 'updated_trust': 0.9929},
                {'iteration': 10, 'accuracy': 0.9649, 'updated_trust': 0.9930},
                {'iteration': 11, 'accuracy': 0.9649, 'updated_trust': 0.9930}
            ],
            'convergence_achieved': True,
            'final_accuracy': 0.964912
        },
        'train_results': {
            'train_accuracy': 0.671
        },
        'presence_stats': {},
        'permanence_stats': {},
        'acceptable_ranges': {
            'accuracy': 0.95,
            'trust': 0.85,
            'entropy': 1.5
        }
    }
    
    logger.info("🧪 Testing Dashboard Display with Mock Results")
    logger.info("=" * 50)
    
    # Test the display logic
    accuracy = mock_results.get('accuracy', 0.0)
    accuracy_ok = mock_results.get('accuracy_ok', False)
    
    logger.info(f"📊 Mock Results:")
    logger.info(f"   Accuracy: {accuracy:.6f}")
    logger.info(f"   Accuracy OK: {accuracy_ok}")
    logger.info(f"   Trust Score: {mock_results.get('trust_score', 0.0):.6f}")
    logger.info(f"   Entropy: {mock_results.get('entropy', 0.0):.6f}")
    logger.info(f"   Block Count: {mock_results.get('block_count', 0)}")
    
    # Test the display formatting
    accuracy_display = f"{accuracy:.3f}"
    accuracy_delta = f"{accuracy - 0.95:.3f}" if accuracy > 0.95 else f"{accuracy - 0.95:.3f}"
    
    logger.info(f"\n📊 Display Formatting:")
    logger.info(f"   Accuracy Display: {accuracy_display}")
    logger.info(f"   Accuracy Delta: {accuracy_delta}")
    logger.info(f"   Delta Color: {'normal' if accuracy_ok else 'inverse'}")
    
    # Test threshold validation
    logger.info(f"\n🔍 Threshold Validation:")
    logger.info(f"   Accuracy ≥ 0.95: {accuracy >= 0.95} ✅" if accuracy >= 0.95 else f"   Accuracy ≥ 0.95: {accuracy >= 0.95} ❌")
    logger.info(f"   Trust ≥ 0.85: {mock_results.get('trust_score', 0.0) >= 0.85} ✅" if mock_results.get('trust_score', 0.0) >= 0.85 else f"   Trust ≥ 0.85: {mock_results.get('trust_score', 0.0) >= 0.85} ❌")
    logger.info(f"   Entropy ≤ 1.5: {mock_results.get('entropy', 0.0) <= 1.5} ✅" if mock_results.get('entropy', 0.0) <= 1.5 else f"   Entropy ≤ 1.5: {mock_results.get('entropy', 0.0) <= 1.5} ❌")
    
    return mock_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dashboard_display() 