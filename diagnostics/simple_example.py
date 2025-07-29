"""
Simple example of SREE Per-Block Diagnostics System
Demonstrates basic functionality without external dependencies.
"""

import numpy as np
from datetime import datetime

# Import diagnostics components
from diagnostics import (
    DiagnosticsConfig,
    DiagnosticsService,
    get_diagnostics_service,
    get_diagnostics_hooks,
    get_diagnostics_api
)


def enable_diagnostics():
    """Enable diagnostics system."""
    # Create configuration
    config = DiagnosticsConfig(
        enabled=True,           # Enable diagnostics
        persist=False,          # Don't persist to disk for this example
        max_rows_per_block=1000  # Limit rows per block
    )
    
    # Set up service with configuration
    service = DiagnosticsService(config)
    
    # Set as global service
    from diagnostics.service import set_diagnostics_service
    set_diagnostics_service(service)
    
    print("✅ Diagnostics enabled with configuration:")
    print(f"   - Enabled: {config.enabled}")
    print(f"   - Persist: {config.persist}")
    print(f"   - Max rows per block: {config.max_rows_per_block}")


def simulate_block_processing():
    """Simulate block processing with diagnostics."""
    print("\n🚀 Simulating block processing with diagnostics...")
    
    hooks = get_diagnostics_hooks()
    
    # Simulate multiple blocks
    for block_index in range(3):
        run_id = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        block_size = 50
        
        print(f"\n📦 Processing Block {block_index}...")
        
        # Emit block start
        hooks.emit_block_start(run_id, block_index, block_size)
        
        # Simulate row processing
        for row_id in range(block_size):
            # Generate some realistic validation scores
            vq = np.random.normal(0.7, 0.2)  # Entropy score
            vb = np.random.normal(0.6, 0.15)  # Hash score
            vl = np.random.normal(0.8, 0.1)   # Logic score
            
            # Determine action based on scores
            if vq > 1.0:  # High entropy
                action = "DOWN_WEIGHTED"
                weight_delta = -0.25
                reason = f"High entropy detected ({vq:.3f} > 1.0)"
            elif vb < 0.3:  # Low hash score
                action = "FLAGGED"
                weight_delta = None
                reason = f"Low hash validation ({vb:.3f} < 0.3)"
            elif vl < 0.5:  # Low logic score
                action = "DOWN_WEIGHTED"
                weight_delta = -0.15
                reason = f"Low logic validation ({vl:.3f} < 0.5)"
            else:
                action = "RETAINED"
                weight_delta = None
                reason = "All validations passed"
            
            # Emit row validation
            hooks.emit_row_validation(
                run_id=run_id,
                block_index=block_index,
                row_id=row_id,
                vq=vq,
                vb=vb,
                vl=vl,
                action=action,
                weight_delta=weight_delta,
                reason=reason
            )
        
        # Emit block end
        summary = hooks.emit_block_end(run_id, block_index)
        
        print(f"   ✅ Block {block_index} completed:")
        print(f"      - Affected rows: {summary['affected_rows']}")
        print(f"      - Actions: {summary['actions_count']}")
        print(f"      - Avg V_q: {summary.get('avg_vq', 'N/A'):.3f}")
        print(f"      - Avg V_b: {summary.get('avg_vb', 'N/A'):.3f}")
        print(f"      - Avg V_l: {summary.get('avg_vl', 'N/A'):.3f}")
    
    return run_id


def examine_diagnostics(run_id):
    """Examine the collected diagnostics."""
    print(f"\n🔍 Examining diagnostics for run {run_id}...")
    
    # Get service and API
    service = get_diagnostics_service()
    api = get_diagnostics_api()
    
    # Get statistics
    stats = service.get_stats()
    print(f"📊 Diagnostics Statistics:")
    print(f"   - Total runs: {stats.get('total_runs', 0)}")
    print(f"   - Total diagnostics: {stats.get('total_diagnostics', 0)}")
    print(f"   - Total summaries: {stats.get('total_summaries', 0)}")
    
    # Get block summaries
    summaries = api.get_block_summaries(run_id)
    print(f"\n📋 Block Summaries:")
    for summary in summaries:
        print(f"   Block {summary['block_index']}:")
        print(f"     - Affected rows: {summary['affected_rows']}")
        print(f"     - Actions: {summary['actions_count']}")
        print(f"     - Avg V_q: {summary.get('avg_vq', 'N/A'):.3f}")
        print(f"     - Avg V_b: {summary.get('avg_vb', 'N/A'):.3f}")
        print(f"     - Avg V_l: {summary.get('avg_vl', 'N/A'):.3f}")
    
    # Get detailed diagnostics for first block
    if summaries:
        block_index = summaries[0]['block_index']
        diagnostics = api.get_block_diagnostics(run_id, block_index)
        print(f"\n🔍 Detailed Diagnostics for Block {block_index}:")
        print(f"   - Total diagnostics: {len(diagnostics)}")
        
        # Show first few diagnostics
        for i, diagnostic in enumerate(diagnostics[:5]):
            print(f"   Row {diagnostic['row_id']}:")
            print(f"     - Action: {diagnostic['action']}")
            print(f"     - V_q: {diagnostic.get('vq', 'N/A'):.3f}")
            print(f"     - V_b: {diagnostic.get('vb', 'N/A'):.3f}")
            print(f"     - V_l: {diagnostic.get('vl', 'N/A'):.3f}")
            print(f"     - Weight delta: {diagnostic.get('weight_delta', 'N/A')}")
            print(f"     - Reason: {diagnostic.get('reason', 'N/A')}")
            if i < 4:  # Don't print separator after last item
                print("     ---")


def demonstrate_logic_validation():
    """Demonstrate logic validation with column insights."""
    print("\n🔧 Demonstrating logic validation diagnostics...")
    
    hooks = get_diagnostics_hooks()
    
    # Simulate logic validation
    run_id = "logic_demo_123"
    block_index = 0
    
    # Feature names and values
    feature_names = ["age", "cholesterol", "blood_pressure", "heart_rate"]
    feature_values = np.array([65, 280, 140, 95])  # Some concerning values
    
    # Logic rules
    logic_rules = [
        {"feature_index": 0, "description": "age > 60", "reason": "High risk age"},
        {"feature_index": 1, "description": "cholesterol > 240", "reason": "High cholesterol"},
        {"feature_index": 2, "description": "blood_pressure > 130", "reason": "High blood pressure"},
        {"feature_index": 3, "description": "heart_rate > 90", "reason": "High heart rate"}
    ]
    
    # Validation results (some rules fail)
    validation_results = [True, False, False, True]  # cholesterol and blood pressure fail
    
    # Emit logic validation
    insights = hooks.emit_logic_validation(
        run_id=run_id,
        block_index=block_index,
        row_id=42,
        feature_names=feature_names,
        feature_values=feature_values,
        logic_rules=logic_rules,
        validation_results=validation_results,
        weight_delta=-0.25
    )
    
    print(f"✅ Logic validation diagnostics emitted:")
    print(f"   - Run ID: {run_id}")
    print(f"   - Row ID: 42")
    print(f"   - Column insights: {len(insights)}")
    
    for insight in insights:
        print(f"     - Column: {insight.column}")
        print(f"       Rule: {insight.rule}")
        print(f"       Delta: {insight.delta}")
        print(f"       Reason: {insight.reason}")


def demonstrate_feature_flagging():
    """Demonstrate feature flagging behavior."""
    print("\n🔒 Demonstrating feature flagging...")
    
    # Test with diagnostics disabled
    print("\n📴 Testing with diagnostics DISABLED:")
    config_disabled = DiagnosticsConfig(enabled=False)
    service_disabled = DiagnosticsService(config_disabled)
    
    # These operations should have no effect
    service_disabled.emit_row("test_run", 0, {
        'row_id': 1,
        'action': 'RETAINED',
        'vq': 0.8,
        'vb': 0.7,
        'vl': 0.6
    })
    
    summary_disabled = service_disabled.summarize_block("test_run", 0)
    print(f"   - Diagnostics stored: {len(service_disabled._diagnostics)}")
    print(f"   - Summary affected rows: {summary_disabled.affected_rows}")
    
    # Test with diagnostics enabled
    print("\n📊 Testing with diagnostics ENABLED:")
    config_enabled = DiagnosticsConfig(enabled=True)
    service_enabled = DiagnosticsService(config_enabled)
    
    # These operations should store data
    service_enabled.emit_row("test_run", 0, {
        'row_id': 1,
        'action': 'RETAINED',
        'vq': 0.8,
        'vb': 0.7,
        'vl': 0.6
    })
    
    summary_enabled = service_enabled.summarize_block("test_run", 0)
    print(f"   - Diagnostics stored: {len(service_enabled._diagnostics)}")
    print(f"   - Summary affected rows: {summary_enabled.affected_rows}")


def main():
    """Main example function."""
    print("🔍 SREE Per-Block Diagnostics - Simple Example")
    print("=" * 55)
    
    # Step 1: Enable diagnostics
    enable_diagnostics()
    
    # Step 2: Simulate block processing
    run_id = simulate_block_processing()
    
    # Step 3: Demonstrate logic validation
    demonstrate_logic_validation()
    
    # Step 4: Demonstrate feature flagging
    demonstrate_feature_flagging()
    
    # Step 5: Examine diagnostics
    examine_diagnostics(run_id)
    
    print("\n✅ Example completed!")
    print("\n🎯 Key Features Demonstrated:")
    print("   ✅ Feature flagging (enabled/disabled)")
    print("   ✅ Row-level diagnostics with validation scores")
    print("   ✅ Block summaries with aggregated metrics")
    print("   ✅ Logic validation with column insights")
    print("   ✅ API access to diagnostic data")
    print("   ✅ Zero impact when disabled")
    
    print("\n📚 Next Steps:")
    print("   1. Enable diagnostics in your SREE configuration")
    print("   2. Integrate hooks into your block processing loops")
    print("   3. Access diagnostics via API or dashboard")
    print("   4. Export data for external analysis")


if __name__ == "__main__":
    main() 