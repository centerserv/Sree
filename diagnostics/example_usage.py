"""
Example usage of SREE Per-Block Diagnostics System
Demonstrates how to enable and use the diagnostics system.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import SREE components
from data_loader import DataLoader
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from loop.trust_loop import TrustUpdateLoop

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
        persist=True,           # Persist to disk for auditing
        max_rows_per_block=5000  # Limit rows per block
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


def run_analysis_with_diagnostics():
    """Run SREE analysis with diagnostics enabled."""
    print("\n🚀 Running SREE analysis with diagnostics...")
    
    # Load sample data
    data_loader = DataLoader()
    X, y = data_loader.load_heart_disease_data()
    
    # Initialize validators
    pattern_validator = PatternValidator()
    presence_validator = PresenceValidator()
    permanence_validator = PermanenceValidator()
    logic_validator = LogicValidator()
    
    # Initialize trust loop
    trust_loop = TrustUpdateLoop(validators=[
        pattern_validator,
        presence_validator,
        permanence_validator,
        logic_validator
    ])
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Run PPP loop (this will automatically emit diagnostics)
    print("📊 Running PPP loop...")
    results = trust_loop.run_ppp_loop(X_train, y_train, X_test, y_test)
    
    print(f"✅ Analysis completed!")
    print(f"   - Final accuracy: {results.get('final_accuracy', 0):.3f}")
    print(f"   - Final trust: {results.get('final_trust', 0):.3f}")
    
    return results


def examine_diagnostics():
    """Examine the collected diagnostics."""
    print("\n🔍 Examining diagnostics...")
    
    # Get service and API
    service = get_diagnostics_service()
    api = get_diagnostics_api()
    
    # Get statistics
    stats = service.get_stats()
    print(f"📊 Diagnostics Statistics:")
    print(f"   - Total runs: {stats.get('total_runs', 0)}")
    print(f"   - Total diagnostics: {stats.get('total_diagnostics', 0)}")
    print(f"   - Total summaries: {stats.get('total_summaries', 0)}")
    
    if stats.get('total_runs', 0) == 0:
        print("❌ No diagnostics found. Make sure diagnostics are enabled and analysis was run.")
        return
    
    # Get available runs (in a real scenario, you'd have run IDs)
    # For this example, we'll assume we know the run ID
    run_id = "ppp_loop_20241201_120000"  # Example run ID
    
    try:
        # Get block summaries
        summaries = api.get_block_summaries(run_id)
        print(f"\n📋 Block Summaries for run {run_id}:")
        for summary in summaries:
            print(f"   Block {summary['block_index']}:")
            print(f"     - Affected rows: {summary['affected_rows']}")
            print(f"     - Avg V_q: {summary.get('avg_vq', 'N/A'):.3f}")
            print(f"     - Avg V_b: {summary.get('avg_vb', 'N/A'):.3f}")
            print(f"     - Avg V_l: {summary.get('avg_vl', 'N/A'):.3f}")
            print(f"     - Actions: {summary['actions_count']}")
        
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
        
        # Export to CSV
        print(f"\n📥 Exporting diagnostics to CSV...")
        export_result = api.export_csv(run_id)
        if export_result['success']:
            print(f"✅ Exported to: {export_result['file_path']}")
        else:
            print(f"❌ Export failed: {export_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Error examining diagnostics: {e}")


def demonstrate_hooks():
    """Demonstrate manual use of diagnostics hooks."""
    print("\n🔧 Demonstrating manual diagnostics hooks...")
    
    hooks = get_diagnostics_hooks()
    
    # Simulate block processing
    run_id = "manual_example_123"
    block_index = 0
    
    # Emit block start
    hooks.emit_block_start(run_id, block_index, 100)
    
    # Emit some row validations
    for i in range(5):
        hooks.emit_row_validation(
            run_id=run_id,
            block_index=block_index,
            row_id=i,
            vq=0.8 + i * 0.1,
            vb=0.6 + i * 0.1,
            vl=0.4 + i * 0.1,
            action="DOWN_WEIGHTED" if i < 2 else "RETAINED",
            weight_delta=-0.25 if i < 2 else None,
            reason=f"Row {i} processing"
        )
    
    # Emit block end
    summary = hooks.emit_block_end(run_id, block_index)
    
    print(f"✅ Manual diagnostics emitted:")
    print(f"   - Run ID: {run_id}")
    print(f"   - Block: {block_index}")
    print(f"   - Affected rows: {summary['affected_rows']}")
    print(f"   - Actions: {summary['actions_count']}")


def main():
    """Main example function."""
    print("🔍 SREE Per-Block Diagnostics Example")
    print("=" * 50)
    
    # Step 1: Enable diagnostics
    enable_diagnostics()
    
    # Step 2: Run analysis (this will automatically emit diagnostics)
    # Uncomment the next line to run actual analysis
    # results = run_analysis_with_diagnostics()
    
    # Step 3: Demonstrate manual hooks
    demonstrate_hooks()
    
    # Step 4: Examine diagnostics
    examine_diagnostics()
    
    print("\n✅ Example completed!")
    print("\nTo enable diagnostics in your own code:")
    print("1. Import diagnostics components")
    print("2. Create DiagnosticsConfig with enabled=True")
    print("3. Set up DiagnosticsService with the config")
    print("4. Use hooks in your block processing loops")
    print("5. Access diagnostics via API or service methods")


if __name__ == "__main__":
    main() 