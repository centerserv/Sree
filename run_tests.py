#!/usr/bin/env python3
"""
SREE Test Runner
Comprehensive test runner for the SREE system.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --fault-injection  # Run only fault injection tests
    python run_tests.py --real-datasets    # Run only real dataset tests
    python run_tests.py --comprehensive    # Run comprehensive test suite
    python run_tests.py --verbose          # Run with verbose output
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def run_pytest(test_pattern, verbose=False):
    """Run pytest with specified pattern."""
    import sys
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    
    if test_pattern:
        cmd.append(f"-k={test_pattern}")
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "--tb=short",
        "--strict-markers"
    ])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="SREE Test Runner")
    parser.add_argument("--unit", action="store_true", 
                       help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", 
                       help="Run only integration tests")
    parser.add_argument("--fault-injection", action="store_true", 
                       help="Run only fault injection tests")
    parser.add_argument("--real-datasets", action="store_true", 
                       help="Run only real dataset tests")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive test suite")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    print("🧪 SREE Test Runner")
    print("=" * 50)
    
    # Determine which tests to run
    if args.unit:
        test_pattern = "test_pattern_layer or test_presence_layer or test_permanence_layer or test_logic_layer"
        print("Running UNIT TESTS...")
    elif args.integration:
        test_pattern = "test_trust_loop or test_setup"
        print("Running INTEGRATION TESTS...")
    elif args.fault_injection:
        test_pattern = "test_fault_injection"
        print("Running FAULT INJECTION TESTS...")
    elif args.real_datasets:
        test_pattern = "test_real_datasets"
        print("Running REAL DATASET TESTS...")
    elif args.comprehensive:
        test_pattern = "comprehensive_test"
        print("Running COMPREHENSIVE TEST SUITE...")
    else:
        test_pattern = None
        print("Running ALL TESTS...")
    
    # Run the tests
    exit_code = run_pytest(test_pattern, args.verbose)
    
    print("=" * 50)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 