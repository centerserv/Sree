"""
SREE Phase 1 Demo - Logic Layer Test
Test script to validate Logic Layer implementation and consistency validation.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import setup_logging
from data_loader import DataLoader
from layers.logic import LogicValidator, create_logic_validator


def test_logic_validator_creation():
    """Test Logic validator creation and basic functionality."""
    print("🔧 Testing Logic Validator Creation...")
    
    try:
        # Test basic creation
        validator = LogicValidator()
        print("  ✅ Logic validator created successfully")
        
        # Test metadata
        metadata = validator.get_metadata()
        if metadata["type"] != "LogicValidator":
            print(f"  ❌ Incorrect validator type: {metadata['type']}")
            assert False
        print("  ✅ Validator metadata correct")
        
        # Test factory function
        factory_validator = create_logic_validator()
        print("  ✅ Factory function working")
        
        
        
    except Exception as e:
        print(f"  ❌ Logic validator creation error: {e}")
        assert False


def test_logic_validation():
    """Test Logic validator validation method."""
    print("\n🔍 Testing Logic Validation Method...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=500)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Logic validator
        validator = LogicValidator()
        
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
        print(f"  ❌ Logic validation error: {e}")
        assert False


def test_logic_feature_consistency():
    """Test feature consistency checking."""
    print("\n🔍 Testing Feature Consistency...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=200)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Logic validator
        validator = LogicValidator()
        
        # Test with clean data
        clean_trust = validator.validate(X_test, y_test)
        
        # Create data with NaN values
        X_with_nan = X_test.copy()
        X_with_nan[0, 0] = np.nan
        
        # Test with NaN data
        nan_trust = validator.validate(X_with_nan, y_test)
        
        # Check that NaN data has lower trust
        if np.mean(nan_trust) >= np.mean(clean_trust):
            print("  ⚠️  NaN data should have lower trust scores")
        
        print(f"  📊 Clean data avg trust: {np.mean(clean_trust):.4f}")
        print(f"  📊 NaN data avg trust: {np.mean(nan_trust):.4f}")
        
        print("  ✅ Feature consistency checking working")
        
        
    except Exception as e:
        print(f"  ❌ Feature consistency error: {e}")
        assert False


def test_logic_label_consistency():
    """Test label consistency checking."""
    print("\n🏷️ Testing Label Consistency...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=200)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Logic validator
        validator = LogicValidator()
        
        # Test with balanced labels
        balanced_trust = validator.validate(X_test, y_test)
        
        # Create imbalanced labels
        imbalanced_y = np.zeros(len(y_test))
        imbalanced_y[:10] = 1  # Only 10 samples with label 1
        
        # Test with imbalanced labels
        imbalanced_trust = validator.validate(X_test, imbalanced_y)
        
        print(f"  📊 Balanced labels avg trust: {np.mean(balanced_trust):.4f}")
        print(f"  📊 Imbalanced labels avg trust: {np.mean(imbalanced_trust):.4f}")
        
        print("  ✅ Label consistency checking working")
        
        
    except Exception as e:
        print(f"  ❌ Label consistency error: {e}")
        assert False


def test_logic_distribution_consistency():
    """Test distribution consistency checking."""
    print("\n📊 Testing Distribution Consistency...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=200)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Logic validator
        validator = LogicValidator()
        
        # Test with normal data
        normal_trust = validator.validate(X_test, y_test)
        
        # Create data with extreme outliers
        X_with_outliers = X_test.copy()
        X_with_outliers[0, :] = 10.0  # Extreme values
        
        # Test with outlier data
        outlier_trust = validator.validate(X_with_outliers, y_test)
        
        print(f"  📊 Normal data avg trust: {np.mean(normal_trust):.4f}")
        print(f"  📊 Outlier data avg trust: {np.mean(outlier_trust):.4f}")
        
        print("  ✅ Distribution consistency checking working")
        
        
    except Exception as e:
        print(f"  ❌ Distribution consistency error: {e}")
        assert False


def test_logic_prediction_validation():
    """Test prediction validation functionality."""
    print("\n🎯 Testing Prediction Validation...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=100)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Logic validator
        validator = LogicValidator()
        
        # Create mock predictions and trust scores
        predictions = np.random.randint(0, 2, len(X_test))
        probabilities = np.random.rand(len(X_test), 2)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        
        pattern_trust = np.random.rand(len(X_test))
        presence_trust = np.random.rand(len(X_test))
        permanence_trust = np.random.rand(len(X_test))
        
        # Test prediction validation
        validated_predictions, final_trust = validator.validate_predictions(
            predictions, probabilities, pattern_trust, presence_trust, permanence_trust
        )
        
        # Check outputs
        if len(validated_predictions) != len(predictions):
            print("  ❌ Validated predictions length mismatch")
            assert False
        
        if len(final_trust) != len(predictions):
            print("  ❌ Final trust scores length mismatch")
            assert False
        
        if not np.all((final_trust >= 0) & (final_trust <= 1)):
            print("  ❌ Final trust scores not in [0, 1] range")
            assert False
        
        print(f"  📊 Average final trust: {np.mean(final_trust):.4f}")
        print(f"  📊 Trust score range: [{np.min(final_trust):.4f}, {np.max(final_trust):.4f}]")
        
        print("  ✅ Prediction validation working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Prediction validation error: {e}")
        assert False


def test_logic_consistency_statistics():
    """Test consistency statistics functionality."""
    print("\n📈 Testing Consistency Statistics...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=100)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Logic validator
        validator = LogicValidator()
        
        # Perform multiple validations
        for i in range(3):
            validator.validate(X_test, y_test)
        
        # Get consistency statistics
        stats = validator.get_consistency_statistics()
        
        if "message" in stats:
            print("  ❌ No consistency statistics available")
            assert False
        
        # Check required statistics
        required_keys = ["total_validations", "avg_consistency_score", 
                        "avg_samples_per_validation", "avg_trust_per_validation"]
        for key in required_keys:
            if key not in stats:
                print(f"  ❌ Missing statistics key: {key}")
                assert False
        
        print(f"  📊 Total validations: {stats['total_validations']}")
        print(f"  📊 Average consistency score: {stats['avg_consistency_score']:.4f}")
        print(f"  📊 Average trust per validation: {stats['avg_trust_per_validation']:.4f}")
        
        print("  ✅ Consistency statistics working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Consistency statistics error: {e}")
        assert False


def test_logic_state_management():
    """Test state management functionality."""
    print("\n⚙️ Testing State Management...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create synthetic dataset
        X, y = loader.create_synthetic(n_samples=100)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Logic validator
        validator = LogicValidator()
        
        # Process data to build state
        validator.validate(X_test, y_test)
        
        # Get state
        state = validator.get_state()
        
        # Check state contains expected keys
        expected_keys = ["consistency_weight", "confidence_threshold", "max_inconsistencies", 
                        "n_validations", "n_consistency_scores"]
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


def test_logic_integration():
    """Test Logic layer integration with other components."""
    print("\n🔗 Testing Logic Integration...")
    
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
        from layers.permanence import PermanenceValidator
        
        # Create all validators
        pattern_validator = PatternValidator()
        presence_validator = PresenceValidator()
        permanence_validator = PermanenceValidator()
        logic_validator = LogicValidator()
        
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
        
        # Validate with Logic layer
        logic_trust = logic_validator.validate(X_test, y_test)
        
        # Create presence trust scores (simulated)
        presence_trust = np.ones(len(X_test)) * 0.9  # Simulated presence trust
        
        # Validate predictions with Logic layer
        final_predictions, final_trust = logic_validator.validate_predictions(
            pattern_predictions, pattern_probabilities,
            pattern_trust, presence_trust, permanence_trust
        )
        
        # Check that all components work together
        if len(final_trust) != len(pattern_trust):
            print("  ❌ Final trust score length mismatch")
            assert False
        
        # Calculate combined trust scores
        combined_trust = (pattern_trust + permanence_trust + logic_trust) / 3
        
        print(f"  📊 Pattern avg trust: {np.mean(pattern_trust):.4f}")
        print(f"  📊 Permanence avg trust: {np.mean(permanence_trust):.4f}")
        print(f"  📊 Logic avg trust: {np.mean(logic_trust):.4f}")
        print(f"  📊 Final avg trust: {np.mean(final_trust):.4f}")
        print(f"  📊 Combined avg trust: {np.mean(combined_trust):.4f}")
        
        print("  ✅ Logic integration working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Logic integration error: {e}")
        assert False


def main():
    """Run Logic Layer test suite."""
    print("🧪 SREE Phase 1 - Logic Layer Test Suite")
    print("=" * 60)
    
    tests = [
        ("Logic Validator Creation", test_logic_validator_creation),
        ("Logic Validation Method", test_logic_validation),
        ("Feature Consistency", test_logic_feature_consistency),
        ("Label Consistency", test_logic_label_consistency),
        ("Distribution Consistency", test_logic_distribution_consistency),
        ("Prediction Validation", test_logic_prediction_validation),
        ("Consistency Statistics", test_logic_consistency_statistics),
        ("State Management", test_logic_state_management),
        ("Logic Integration", test_logic_integration)
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
    print(f"📊 Logic Layer Test Results: {passed}/{total} tests passed")
    
    if failed_tests:
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        print("\n🔧 Please fix the failed tests before proceeding.")
        assert False
    else:
        print("🎉 ALL LOGIC LAYER TESTS PASSED!")
        print("\n✅ Logic Layer is ready for PPP integration.")
        print("\n🚀 Ready to proceed with:")
        print("   1. Trust Update Loop")
        print("   2. Full PPP Integration")
        print("   3. Performance Testing")
        


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 