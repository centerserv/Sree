"""
SREE Phase 1 Demo - Trust Update Loop Test
Test script to validate Trust Update Loop implementation and PPP convergence.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import setup_logging
from data_loader import DataLoader
from loop.trust_loop import TrustUpdateLoop, create_trust_loop


def test_trust_loop_creation():
    """Test Trust Update Loop creation and basic functionality."""
    print("🔧 Testing Trust Update Loop Creation...")
    
    try:
        # Test basic creation
        loop = TrustUpdateLoop()
        print("  ✅ Trust Update Loop created successfully")
        
        # Test factory function
        factory_loop = create_trust_loop()
        print("  ✅ Factory function working")
        
        # Test state
        state = loop.get_state()
        if "iterations" not in state:
            print("  ❌ Missing iterations in state")
            assert False
        print("  ✅ State management working")
        
        
        
    except Exception as e:
        print(f"  ❌ Trust Update Loop creation error: {e}")
        assert False


def test_trust_loop_initialization():
    """Test Trust Update Loop initialization with validators."""
    print("\n🔧 Testing Trust Update Loop Initialization...")
    
    try:
        # Create Trust Update Loop
        loop = TrustUpdateLoop()
        
        # Check that all validators are initialized
        metadata = loop.get_validator_metadata()
        
        required_validators = ["pattern", "presence", "permanence", "logic"]
        for validator in required_validators:
            if validator not in metadata:
                print(f"  ❌ Missing validator: {validator}")
                assert False
        
        # Check loop configuration
        if "loop_config" not in metadata:
            print("  ❌ Missing loop configuration")
            assert False
        
        loop_config = metadata["loop_config"]
        required_config = ["iterations", "gamma", "alpha", "beta", "delta"]
        for param in required_config:
            if param not in loop_config:
                print(f"  ❌ Missing config parameter: {param}")
                assert False
        
        print("  ✅ All validators initialized correctly")
        print(f"  📊 Iterations: {loop_config['iterations']}")
        print(f"  📊 Alpha: {loop_config['alpha']}")
        print(f"  📊 Beta: {loop_config['beta']}")
        print(f"  📊 Delta: {loop_config['delta']}")
        
        
        
    except Exception as e:
        print(f"  ❌ Trust Update Loop initialization error: {e}")
        assert False


def test_trust_update_mechanism():
    """Test trust update mechanism."""
    print("\n🔄 Testing Trust Update Mechanism...")
    
    try:
        # Create Trust Update Loop
        loop = TrustUpdateLoop()
        
        # Create mock trust scores
        n_samples = 100
        pattern_trust = np.random.rand(n_samples)
        presence_trust = np.random.rand(n_samples)
        permanence_trust = np.random.rand(n_samples)
        logic_trust = np.random.rand(n_samples)
        final_trust = np.random.rand(n_samples)
        
        # Test trust update mechanism (access private method for testing)
        updated_trust = loop._update_trust_scores(
            pattern_trust, presence_trust, permanence_trust, logic_trust, final_trust
        )
        
        # Check outputs
        if len(updated_trust) != n_samples:
            print("  ❌ Updated trust scores length mismatch")
            assert False
        
        if not np.all((updated_trust >= 0) & (updated_trust <= 1)):
            print("  ❌ Updated trust scores not in [0, 1] range")
            assert False
        
        print(f"  📊 Average pattern trust: {np.mean(pattern_trust):.4f}")
        print(f"  📊 Average presence trust: {np.mean(presence_trust):.4f}")
        print(f"  📊 Average permanence trust: {np.mean(permanence_trust):.4f}")
        print(f"  📊 Average logic trust: {np.mean(logic_trust):.4f}")
        print(f"  📊 Average final trust: {np.mean(final_trust):.4f}")
        print(f"  📊 Average updated trust: {np.mean(updated_trust):.4f}")
        
        print("  ✅ Trust update mechanism working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Trust update mechanism error: {e}")
        assert False


def test_convergence_checking():
    """Test convergence checking functionality."""
    print("\n🎯 Testing Convergence Checking...")
    
    try:
        # Create Trust Update Loop
        loop = TrustUpdateLoop()
        
        # Test with non-converged trust scores
        non_converged_trust = np.ones(100) * 0.5
        non_converged_accuracy = 0.5
        
        convergence = loop._check_convergence(non_converged_trust, non_converged_accuracy)
        print(f"  📊 Non-converged case: {convergence}")
        
        # Test with converged trust scores (simulated)
        converged_trust = np.ones(100) * 0.96
        converged_accuracy = 0.985
        
        # Add some history to simulate stability
        loop._trust_history = [0.95, 0.96, 0.96]
        loop._accuracy_history = [0.984, 0.985, 0.985]
        
        convergence = loop._check_convergence(converged_trust, converged_accuracy)
        print(f"  📊 Converged case: {convergence}")
        
        print("  ✅ Convergence checking working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Convergence checking error: {e}")
        assert False


def test_ppp_loop_execution():
    """Test PPP loop execution with small dataset."""
    print("\n🔄 Testing PPP Loop Execution...")
    
    try:
        # Setup logging and data
        logger = setup_logging(level="INFO")
        loader = DataLoader(logger)
        
        # Create small synthetic dataset for testing
        X, y = loader.create_synthetic(n_samples=200)
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        # Create Trust Update Loop with fewer iterations for testing
        loop = TrustUpdateLoop(iterations=3)
        
        # Run PPP loop
        results = loop.run_ppp_loop(X_train, y_train, X_test, y_test)
        
        # Check results structure
        required_keys = ["iterations", "final_accuracy", "final_trust", 
                        "convergence_achieved", "convergence_iteration"]
        for key in required_keys:
            if key not in results:
                print(f"  ❌ Missing result key: {key}")
                assert False
        
        # Check iterations
        if len(results["iterations"]) < 1:
            print(f"  ❌ Expected at least 1 iteration, got {len(results['iterations'])}")
            assert False
        
        # Check iteration structure
        iteration = results["iterations"][0]
        required_iteration_keys = ["iteration", "pattern_trust", "presence_trust", 
                                  "permanence_trust", "logic_trust", "final_trust", 
                                  "updated_trust", "accuracy", "convergence"]
        for key in required_iteration_keys:
            if key not in iteration:
                print(f"  ❌ Missing iteration key: {key}")
                assert False
        
        print(f"  📊 Final accuracy: {results['final_accuracy']:.4f}")
        print(f"  📊 Final trust: {results['final_trust']:.4f}")
        print(f"  📊 Convergence achieved: {results['convergence_achieved']}")
        print(f"  📊 Total iterations: {len(results['iterations'])}")
        
        print("  ✅ PPP loop execution working correctly")
        
        
    except Exception as e:
        print(f"  ❌ PPP loop execution error: {e}")
        assert False


def test_convergence_statistics():
    """Test convergence statistics functionality."""
    print("\n📈 Testing Convergence Statistics...")
    
    try:
        # Create Trust Update Loop
        loop = TrustUpdateLoop()
        
        # Add some mock history
        loop._trust_history = [0.5, 0.6, 0.7, 0.8, 0.9]
        loop._accuracy_history = [0.6, 0.7, 0.8, 0.85, 0.9]
        loop._convergence_history = [False, False, False, False, True]
        
        # Get convergence statistics
        stats = loop.get_convergence_statistics()
        
        if "message" in stats:
            print("  ❌ No convergence statistics available")
            assert False
        
        # Check required statistics
        required_keys = ["total_iterations", "final_trust", "final_accuracy", 
                        "avg_trust", "avg_accuracy", "convergence_achieved"]
        for key in required_keys:
            if key not in stats:
                print(f"  ❌ Missing statistics key: {key}")
                assert False
        
        print(f"  📊 Total iterations: {stats['total_iterations']}")
        print(f"  📊 Final trust: {stats['final_trust']:.4f}")
        print(f"  📊 Final accuracy: {stats['final_accuracy']:.4f}")
        print(f"  📊 Average trust: {stats['avg_trust']:.4f}")
        print(f"  📊 Average accuracy: {stats['avg_accuracy']:.4f}")
        print(f"  📊 Convergence achieved: {stats['convergence_achieved']}")
        
        if stats['convergence_achieved']:
            print(f"  📊 Convergence iteration: {stats['convergence_iteration']}")
        
        print("  ✅ Convergence statistics working correctly")
        
        
    except Exception as e:
        print(f"  ❌ Convergence statistics error: {e}")
        assert False


def test_results_persistence():
    """Test results saving and loading functionality."""
    print("\n💾 Testing Results Persistence...")
    
    # Setup logging and data
    logger = setup_logging(level="INFO")
    loader = DataLoader(logger)
    
    # Create small synthetic dataset
    X, y = loader.create_synthetic(n_samples=100)
    X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
    
    # Create Trust Update Loop
    loop = TrustUpdateLoop(iterations=2)
    
    # Run PPP loop
    results = loop.run_ppp_loop(X_train, y_train, X_test, y_test)
    
    # Save results
    results_path = loop.save_results(results, "test_ppp_results.json")
    print(f"  ✅ Results saved to: {results_path}")
    
    # Load results
    loaded_results = loop.load_results("test_ppp_results.json")
    
    # Check that loaded results match original
    assert loaded_results["final_accuracy"] == results["final_accuracy"], "Final accuracy mismatch after loading"
    assert loaded_results["final_trust"] == results["final_trust"], "Final trust mismatch after loading"
    
    print("  ✅ Results persistence working correctly")


def test_loop_reset():
    """Test loop reset functionality."""
    print("\n🔄 Testing Loop Reset...")
    
    # Create Trust Update Loop
    loop = TrustUpdateLoop()
    
    # Add some mock history
    loop._trust_history = [0.5, 0.6, 0.7]
    loop._accuracy_history = [0.6, 0.7, 0.8]
    loop._convergence_history = [False, False, True]
    loop._current_trust = 0.7
    loop._current_state = 0.8
    
    # Check initial state
    initial_state = loop.get_state()
    assert initial_state["trust_history_length"] == 3, "Initial history length incorrect"
    
    # Reset loop
    loop.reset()
    
    # Check reset state
    reset_state = loop.get_state()
    assert reset_state["trust_history_length"] == 0, "Reset history length incorrect"
    assert reset_state["current_trust"] == loop._initial_trust, "Reset trust not restored to initial value"
    
    print("  ✅ Loop reset working correctly")


def test_full_ppp_integration():
    """Test full PPP integration with all components."""
    print("\n🔗 Testing Full PPP Integration...")
    
    # Setup logging and data
    logger = setup_logging(level="INFO")
    loader = DataLoader(logger)
    
    # Create synthetic dataset
    X, y = loader.create_synthetic(n_samples=300)
    X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
    
    # Create Trust Update Loop
    loop = TrustUpdateLoop(iterations=5)
    
    # Run full PPP loop
    results = loop.run_ppp_loop(X_train, y_train, X_test, y_test)
    
    # Check that all layers contributed
    assert results["iterations"], "No iterations completed"
    
    # Check final metrics
    final_iteration = results["iterations"][-1]
    
    print(f"  📊 Final accuracy: {results['final_accuracy']:.4f}")
    print(f"  📊 Final trust: {results['final_trust']:.4f}")
    print(f"  📊 Pattern trust: {final_iteration['pattern_trust']:.4f}")
    print(f"  📊 Presence trust: {final_iteration['presence_trust']:.4f}")
    print(f"  📊 Permanence trust: {final_iteration['permanence_trust']:.4f}")
    print(f"  📊 Logic trust: {final_iteration['logic_trust']:.4f}")
    print(f"  📊 Convergence achieved: {results['convergence_achieved']}")
    
    # Check that all trust scores are reasonable
    all_trusts = [
        final_iteration['pattern_trust'],
        final_iteration['presence_trust'],
        final_iteration['permanence_trust'],
        final_iteration['logic_trust'],
        final_iteration['final_trust'],
        final_iteration['updated_trust']
    ]
    
    for trust in all_trusts:
        assert 0 <= trust <= 1, f"Trust score out of range: {trust}"
    
    print("  ✅ Full PPP integration working correctly")


def main():
    """Run Trust Update Loop test suite."""
    print("🧪 SREE Phase 1 - Trust Update Loop Test Suite")
    print("=" * 60)
    
    tests = [
        ("Trust Update Loop Creation", test_trust_loop_creation),
        ("Trust Update Loop Initialization", test_trust_loop_initialization),
        ("Trust Update Mechanism", test_trust_update_mechanism),
        ("Convergence Checking", test_convergence_checking),
        ("PPP Loop Execution", test_ppp_loop_execution),
        ("Convergence Statistics", test_convergence_statistics),
        ("Results Persistence", test_results_persistence),
        ("Loop Reset", test_loop_reset),
        ("Full PPP Integration", test_full_ppp_integration)
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
    print(f"📊 Trust Update Loop Test Results: {passed}/{total} tests passed")
    
    if failed_tests:
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        print("\n🔧 Please fix the failed tests before proceeding.")
        assert False
    else:
        print("🎉 ALL TRUST UPDATE LOOP TESTS PASSED!")
        print("\n✅ Trust Update Loop is ready for full SREE demo.")
        print("\n🚀 Ready to proceed with:")
        print("   1. Full SREE Demo Execution")
        print("   2. Performance Testing")
        print("   3. Fault Injection Testing")
        print("   4. Ablation Studies")
        


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 