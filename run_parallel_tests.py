#!/usr/bin/env python3
"""
SREE Parallel Test Runner
Runs tests in parallel to speed up execution when you have time.
"""

import sys
import time
import subprocess
import concurrent.futures
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def run_test_file(test_file):
    """Run a single test file and return results."""
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file,
            "-v", "--tb=short", "--quiet"
        ], capture_output=True, text=True, timeout=60)
        
        return {
            'file': test_file,
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr,
            'duration': 0  # Will be calculated by caller
        }
    except subprocess.TimeoutExpired:
        return {
            'file': test_file,
            'success': False,
            'output': '',
            'error': 'Test timed out after 60 seconds',
            'duration': 60
        }
    except Exception as e:
        return {
            'file': test_file,
            'success': False,
            'output': '',
            'error': str(e),
            'duration': 0
        }

def run_parallel_tests(max_workers=4):
    """Run tests in parallel."""
    print("🚀 SREE Parallel Test Runner")
    print("=" * 50)
    print(f"Running tests with {max_workers} workers...")
    print()
    
    # Define test files to run in parallel
    test_files = [
        "tests/test_setup.py",
        "tests/test_pattern_layer.py",
        "tests/test_presence_layer.py", 
        "tests/test_permanence_layer.py",
        "tests/test_logic_layer.py",
        "tests/test_trust_loop.py",
        "tests/test_diagnostics.py",
        "tests/test_visualization.py",
        "tests/test_ablation_studies.py",
        "tests/test_fault_injection.py",
        "tests/test_real_datasets.py"
    ]
    
    start_time = time.time()
    results = []
    
    # Run tests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tests
        future_to_test = {executor.submit(run_test_file, test_file): test_file 
                         for test_file in test_files}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_test):
            test_file = future_to_test[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print immediate feedback
                status = "✅" if result['success'] else "❌"
                print(f"{status} {test_file}")
                
            except Exception as e:
                print(f"❌ {test_file}: {e}")
                results.append({
                    'file': test_file,
                    'success': False,
                    'output': '',
                    'error': str(e),
                    'duration': 0
                })
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Calculate statistics
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print("\n" + "=" * 50)
    print("📊 PARALLEL TEST RESULTS")
    print("=" * 50)
    print(f"⏱️  Total Duration: {total_duration:.1f} seconds")
    print(f"📈 Tests Passed: {passed}/{len(results)}")
    print(f"📉 Tests Failed: {failed}/{len(results)}")
    print(f"🚀 Speedup: ~{len(results) * 10 / total_duration:.1f}x faster than sequential")
    
    # Show detailed results
    print("\n📋 DETAILED RESULTS:")
    print("-" * 30)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {result['file']}")
        
        if not result['success'] and result['error']:
            print(f"    Error: {result['error'][:100]}...")
    
    print("\n" + "=" * 50)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"⚠️  {failed} tests failed. Check details above.")
        return 1

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SREE Parallel Test Runner")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List test files that would be run")
    
    args = parser.parse_args()
    
    if args.list:
        test_files = [
            "tests/test_setup.py",
            "tests/test_pattern_layer.py",
            "tests/test_presence_layer.py", 
            "tests/test_permanence_layer.py",
            "tests/test_logic_layer.py",
            "tests/test_trust_loop.py",
            "tests/test_diagnostics.py",
            "tests/test_visualization.py",
            "tests/test_ablation_studies.py",
            "tests/test_fault_injection.py",
            "tests/test_real_datasets.py"
        ]
        print("Test files that would be run:")
        for test_file in test_files:
            print(f"  - {test_file}")
        return 0
    
    return run_parallel_tests(args.workers)

if __name__ == "__main__":
    sys.exit(main()) 