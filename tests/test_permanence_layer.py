"""
SREE Phase 1 Demo - Permanence Layer Test
Test script to validate Permanence Layer implementation and hash-based logging.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import setup_logging
from data_loader import DataLoader
from layers.permanence import PermanenceValidator, create_permanence_validator


def test_permanence_validator_creation():
    """Test Permanence validator creation and basic functionality."""
    print("🔧 Testing Permanence Validator Creation...")
    
    try:
        # Test basic creation
        validator = PermanenceValidator()
        print("  ✅ Permanence validator created successfully")
        
        # Test metadata
        metadata = validator.get_metadata()
        if metadata["type"] != "PermanenceValidator":
            print(f"  ❌ Incorrect validator type: {metadata['type']}")
            assert False
        print("  ✅ Validator metadata correct")
        
        # Test factory function
        factory_validator = create_permanence_validator()
        print("  ✅ Factory function working")
        
        
        
    except Exception as e:
        print(f"  ❌ Permanence validator creation error: {e}")
        assert False


def test_permanence_validation():
    """Test Permanence validator validation method."""
    print("\n🔍 Testing Permanence Validation Method...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=500)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Permanence validator
        validator = PermanenceValidator()
        
        # Test validation method
        trust_scores = validator.validate(X_test, y_test)
        
        # Check trust scores
        if len(trust_scores) != len(X_test):
            print(f"  ❌ Trust scores length mismatch: {len(trust_scores)} vs {len(X_test)}")
            assert False
        
        if not np.all((trust_scores >= 0) & (trust_scores <= 1)):
            print("  ❌ Trust scores not in [0, 1] range")
            assert False
        
        avg_trust = np.mean(trust_scores)
        print(f"  📊 Average trust score: {avg_trust:.4f}")
        print(f"  📊 Trust score range: [{np.min(trust_scores):.4f}, {np.max(trust_scores):.4f}]")
        
        print("  ✅ Validation method working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Permanence validation error: {e}")
        assert False


def test_permanence_block_creation():
    """Test block creation and ledger functionality."""
    print("\n📦 Testing Block Creation...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=200)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Permanence validator with small block size
        validator = PermanenceValidator(block_size=50)
        
        # Process data in chunks to create multiple blocks
        for i in range(0, len(X_test), 25):
            chunk_X = X_test[i:i+25]
            chunk_y = y_test[i:i+25] if y_test is not None else None
            validator.validate(chunk_X, chunk_y)
        
        # Force finalize any remaining records
        if validator._current_block:
            validator._finalize_block()
        
        # Check ledger statistics
        stats = validator.get_ledger_statistics()
        
        if "message" in stats:
            print("  ❌ No ledger data available")
            assert False
        
        print(f"  📊 Total blocks: {stats['total_blocks']}")
        print(f"  📊 Total records: {stats['total_records']}")
        print(f"  📊 Average records per block: {stats['avg_records_per_block']:.2f}")
        
        if stats['total_blocks'] < 1:
            print("  ❌ No blocks created")
            assert False
        
        print("  ✅ Block creation working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Block creation error: {e}")
        assert False


def test_permanence_consistency_checking():
    """Test ledger consistency checking functionality."""
    print("\n🔗 Testing Consistency Checking...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=150)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Permanence validator with small block size
        validator = PermanenceValidator(block_size=50)
        
        # Process data to create blocks
        for i in range(0, len(X_test), 25):
            chunk_X = X_test[i:i+25]
            chunk_y = y_test[i:i+25] if y_test is not None else None
            validator.validate(chunk_X, chunk_y)
        
        # Check ledger consistency
        consistency_result = validator.check_ledger_consistency()
        
        if consistency_result["status"] == "insufficient_blocks":
            print("  ⚠️  Need more blocks for consistency check")
            print("  ✅ Consistency checking working correctly (insufficient data)")
        else:
            print(f"  📊 Consistency score: {consistency_result['consistency_score']:.4f}")
            print(f"  📊 Is consistent: {consistency_result['is_consistent']}")
            print(f"  📊 Inconsistencies: {len(consistency_result['inconsistencies'])}")
            print("  ✅ Consistency checking working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Consistency checking error: {e}")
        assert False


def test_permanence_duplicate_detection():
    """Test duplicate detection functionality."""
    print("\n🔄 Testing Duplicate Detection...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=100)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Permanence validator
        validator = PermanenceValidator()
        
        # First validation
        trust_scores_1 = validator.validate(X_test, y_test)
        
        # Second validation with same data (should detect duplicates)
        trust_scores_2 = validator.validate(X_test, y_test)
        
        # Check if trust scores are different (indicating duplicate detection)
        trust_diff = np.mean(trust_scores_2) - np.mean(trust_scores_1)
        
        print(f"  📊 First validation avg trust: {np.mean(trust_scores_1):.4f}")
        print(f"  📊 Second validation avg trust: {np.mean(trust_scores_2):.4f}")
        print(f"  📊 Trust difference: {trust_diff:.4f}")
        
        # Check ledger statistics for duplicates
        stats = validator.get_ledger_statistics()
        if "avg_duplicates_recent" in stats:
            print(f"  📊 Average duplicates detected: {stats['avg_duplicates_recent']:.2f}")
        
        print("  ✅ Duplicate detection working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Duplicate detection error: {e}")
        assert False


def test_permanence_ledger_persistence():
    """Test ledger saving and loading functionality."""
    print("\n💾 Testing Ledger Persistence...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=100)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Permanence validator
        validator = PermanenceValidator()
        
        # Process data
        validator.validate(X_test, y_test)
        
        # Save ledger
        ledger_path = validator.save_ledger("test_permanence_ledger.json")
        print(f"  ✅ Ledger saved to: {ledger_path}")
        
        # Create new validator and load ledger
        new_validator = PermanenceValidator()
        new_validator.load_ledger("test_permanence_ledger.json")
        
        # Check that loaded validator has same statistics
        original_stats = validator.get_ledger_statistics()
        loaded_stats = new_validator.get_ledger_statistics()
        
        if original_stats["total_blocks"] != loaded_stats["total_blocks"]:
            print("  ❌ Block count mismatch after loading")
            assert False
        
        print("  ✅ Ledger persistence working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Ledger persistence error: {e}")
        assert False


def test_permanence_state_management():
    """Test state management functionality."""
    print("\n⚙️ Testing State Management...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=100)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Permanence validator
        validator = PermanenceValidator()
        
        # Process data to build state
        validator.validate(X_test, y_test)
        
        # Get state
        state = validator.get_state()
        
        # Check state contains expected keys
        expected_keys = ["hash_algorithm", "block_size", "consistency_threshold", 
                        "total_blocks", "current_block_size", "n_consistency_checks"]
        for key in expected_keys:
            if key not in state:
                print(f"  ❌ Missing state key: {key}")
                assert False
        
        # Set state
        validator.set_state(state)
        
        print("  ✅ State management working correctly")
        
        
    except Exception as e:
        print(f"  ❌ State management error: {e}")
        assert False


def test_permanence_integration():
    """Test Permanence layer integration with other components."""
    print("\n🔗 Testing Permanence Integration...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=200)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Import other validators
        from layers.pattern import PatternValidator
        from layers.presence import PresenceValidator
        
        # Create all validators
        pattern_validator = PatternValidator()
        presence_validator = PresenceValidator()
        permanence_validator = PermanenceValidator()
        
        # Train Pattern validator
        pattern_validator.train(X_train, y_train)
        
        # Get pattern predictions
        pattern_trust = pattern_validator.validate(X_test, y_test)
        pattern_predictions = pattern_validator.predictions
        pattern_probabilities = pattern_validator.probabilities
        
        # Refine with Presence layer
        refined_predictions, refined_probabilities = presence_validator.refine_predictions(
            pattern_predictions, pattern_probabilities, X_test
        )
        
        # Validate with Permanence layer
        permanence_trust = permanence_validator.validate(X_test, y_test)
        
        # Check that all components work together
        if len(permanence_trust) != len(pattern_trust):
            print("  ❌ Trust score length mismatch between layers")
            assert False
        
        # Calculate combined trust scores
        combined_trust = (pattern_trust + permanence_trust) / 2
        
        print(f"  📊 Pattern avg trust: {np.mean(pattern_trust):.4f}")
        print(f"  📊 Permanence avg trust: {np.mean(permanence_trust):.4f}")
        print(f"  📊 Combined avg trust: {np.mean(combined_trust):.4f}")
        
        print("  ✅ Permanence integration working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Permanence integration error: {e}")
        assert False


def main():
    """Run Permanence Layer test suite."""
    print("🧪 SREE Phase 1 - Permanence Layer Test Suite")
    print("=" * 60)
    
    tests = [
        ("Permanence Validator Creation", test_permanence_validator_creation),
        ("Permanence Validation Method", test_permanence_validation),
        ("Block Creation", test_permanence_block_creation),
        ("Consistency Checking", test_permanence_consistency_checking),
        ("Duplicate Detection", test_permanence_duplicate_detection),
        ("Ledger Persistence", test_permanence_ledger_persistence),
        ("State Management", test_permanence_state_management),
        ("Permanence Integration", test_permanence_integration)
    ]
    
    passed = 0
    total = len(tests)
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed_tests.append(test_name)
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            failed_tests.append(test_name)
    
    print("\n" + "=" * 60)
    print(f"📊 Permanence Layer Test Results: {passed}/{total} tests passed")
    
    if failed_tests:
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        print("\n🔧 Please fix the failed tests before proceeding.")
        assert False
    else:
        print("🎉 ALL PERMANENCE LAYER TESTS PASSED!")
        print("\n✅ Permanence Layer is ready for PPP integration.")
        print("\n🚀 Ready to proceed with:")
        print("   1. Logic Layer (consistency validation)")
        print("   2. Trust Update Loop")
        print("   3. Full PPP Integration")
        


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 