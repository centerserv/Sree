# SREE Per-Block Diagnostics

Transparent, auditable logs per block for SREE analysis showing validation scores, actions taken, and weight changes.

## Overview

The Per-Block Diagnostics system provides comprehensive transparency into the SREE block refinement process. It tracks:

- **Validation Scores**: V_q (entropy), V_b (hash change), V_l (logic) for each row
- **Actions Taken**: Down-weighted, retained, or flagged rows with reasons
- **Weight Changes**: Deltas applied to row weights
- **Column Insights**: Feature-level details when logic rules fail

## Features

### 🔒 **Feature Flagged**

- **Default**: Disabled (`perBlockDiagnostics.enabled = false`)
- **Zero Impact**: When disabled, no performance impact or side effects
- **Backward Compatible**: Existing functionality unchanged

### 📊 **Comprehensive Logging**

- Row-level diagnostics with timestamps
- Block summaries with aggregated metrics
- Column-level insights for logic validation failures
- Weight delta tracking

### 🚀 **High Performance**

- In-memory storage with optional persistence
- Configurable row limits to prevent memory explosion
- Sampling for large datasets

### 🔌 **Easy Integration**

- Simple hooks for existing code
- REST API endpoints
- Streamlit dashboard integration

## Configuration

### Basic Configuration

```python
from diagnostics import DiagnosticsConfig

# Enable diagnostics
config = DiagnosticsConfig(
    enabled=True,           # Feature flag (default: False)
    persist=False,          # Persist to disk (default: False)
    max_rows_per_block=10000  # Max rows per block (default: 10000)
)
```

### Environment Configuration

Add to your configuration:

```json
{
  "per_block_diagnostics": {
    "enabled": false,
    "persist": false,
    "max_rows_per_block": 10000
  }
}
```

## Usage

### Basic Usage

```python
from diagnostics import get_diagnostics_service, get_diagnostics_hooks

# Get service instance
service = get_diagnostics_service()

# Get hooks for integration
hooks = get_diagnostics_hooks()

# Emit diagnostics
hooks.emit_row_validation(
    run_id="analysis_123",
    block_index=0,
    row_id=42,
    vq=0.8,  # entropy score
    vb=0.6,  # hash score
    vl=0.4,  # logic score
    action="DOWN_WEIGHTED",
    weight_delta=-0.25,
    reason="High entropy detected"
)

# Get block summary
summary = service.summarize_block("analysis_123", 0)
print(f"Affected rows: {summary.affected_rows}")
```

### Integration with Existing Code

The diagnostics system integrates seamlessly with existing SREE code:

```python
# In your block processing loop
def process_block(X_block, y_block, block_index):
    # ... existing processing ...

    # Emit diagnostics for each row
    for i, row in enumerate(X_block):
        hooks.emit_row_validation(
            run_id=run_id,
            block_index=block_index,
            row_id=i,
            vq=entropy_scores[i],
            vb=hash_scores[i],
            vl=logic_scores[i],
            action=decisions[i],
            weight_delta=weight_changes[i],
            reason=reasons[i]
        )

    # Emit block summary
    summary = hooks.emit_block_end(run_id, block_index)
```

### API Access

```python
from diagnostics import get_diagnostics_api

api = get_diagnostics_api()

# Get block summaries
summaries = api.get_block_summaries("analysis_123")

# Get detailed diagnostics
diagnostics = api.get_block_diagnostics("analysis_123", 0)

# Export to CSV
result = api.export_csv("analysis_123")

# Get statistics
stats = api.get_stats()
```

## Data Models

### BlockRowDiagnostic

```python
@dataclass
class BlockRowDiagnostic:
    run_id: str                    # Unique analysis run ID
    block_index: int               # Block index
    row_id: Union[str, int]        # Row identifier
    action: DiagnosticAction       # DOWN_WEIGHTED, RETAINED, FLAGGED
    timestamp: str                 # ISO timestamp

    # Validation scores (optional)
    vq: Optional[float]            # Entropy score
    vb: Optional[float]            # Hash/change score
    vl: Optional[float]            # Logic score

    # Action details
    weight_delta: Optional[float]  # Weight change applied
    columns: Optional[List[ColumnInsight]]  # Column-level insights
    reason: Optional[str]          # Explanation for auditors
```

### BlockDiagnosticSummary

```python
@dataclass
class BlockDiagnosticSummary:
    run_id: str                    # Unique analysis run ID
    block_index: int               # Block index
    affected_rows: int             # Number of affected rows

    # Average validation scores
    avg_vq: Optional[float]        # Average entropy
    avg_vb: Optional[float]        # Average hash score
    avg_vl: Optional[float]        # Average logic score

    # Action counts
    actions_count: Dict[DiagnosticAction, int]  # Count by action type
```

### ColumnInsight

```python
@dataclass
class ColumnInsight:
    column: str                    # Feature/column name
    rule: Optional[str]            # Logic rule (e.g., "cholesterol > 240")
    delta: Optional[float]         # Weight delta for this column
    reason: Optional[str]          # Explanation
```

## API Endpoints

### REST API

When integrated with Flask:

```python
from diagnostics import create_diagnostics_routes

app = Flask(__name__)
create_diagnostics_routes(app)
```

Available endpoints:

- `GET /sree/diagnostics/:runId/blocks` → BlockDiagnosticSummary[]
- `GET /sree/diagnostics/:runId/block/:index` → BlockRowDiagnostic[]
- `GET /sree/diagnostics/:runId/export.csv` → CSV export
- `GET /sree/diagnostics/stats` → Statistics
- `DELETE /sree/diagnostics/:runId` → Clear run data

### Streamlit Integration

```python
from diagnostics import create_streamlit_diagnostics_section

# In your Streamlit app
diagnostics_section = create_streamlit_diagnostics_section()
diagnostics_section()
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_diagnostics.py -v
```

Tests cover:

- ✅ Feature flag behavior (on/off)
- ✅ Data correctness and serialization
- ✅ API endpoints
- ✅ Integration hooks
- ✅ Edge cases and error handling
- ✅ Performance limits

## Performance Considerations

### Memory Usage

- **Default limit**: 10,000 rows per block
- **Sampling**: Automatic sampling when limits exceeded
- **Cleanup**: Manual cleanup via `clear_run()`

### Storage

- **In-memory**: Default storage (fastest)
- **Persistence**: Optional disk storage for auditing
- **Export**: CSV export for external analysis

### Impact on Core Metrics

- **Zero impact**: Core SREE metrics unchanged
- **Logging only**: Diagnostics are observational
- **No interference**: Existing convergence logic unaffected

## Migration Guide

### Enabling Diagnostics

1. **Update Configuration**:

   ```python
   # In config.py
   PER_BLOCK_DIAGNOSTICS_CONFIG = {
       "enabled": True,  # Enable feature
       "persist": False,  # Optional persistence
       "max_rows_per_block": 10000
   }
   ```

2. **Verify Integration**:

   ```python
   # Check that hooks are working
   from diagnostics import get_diagnostics_hooks
   hooks = get_diagnostics_hooks()
   print(f"Diagnostics enabled: {hooks.service.config.enabled}")
   ```

3. **Test with Small Dataset**:

   ```python
   # Run analysis and check diagnostics
   # ... run your analysis ...

   from diagnostics import get_diagnostics_service
   service = get_diagnostics_service()
   stats = service.get_stats()
   print(f"Diagnostics collected: {stats['total_diagnostics']}")
   ```

### Disabling Diagnostics

Simply set `enabled: false` in configuration. No code changes required.

## Troubleshooting

### Common Issues

1. **No diagnostics appearing**:

   - Check if feature is enabled: `service.config.enabled`
   - Verify hooks are being called
   - Check for import errors

2. **Memory issues**:

   - Reduce `max_rows_per_block`
   - Enable sampling for large datasets
   - Clear old runs: `service.clear_run(run_id)`

3. **API errors**:
   - Verify diagnostics are enabled
   - Check run_id exists
   - Ensure proper error handling

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('diagnostics').setLevel(logging.DEBUG)
```

## Contributing

When adding new diagnostic features:

1. **Maintain backward compatibility**
2. **Add comprehensive tests**
3. **Update documentation**
4. **Follow existing patterns**

## License

Part of the SREE project. See main project license.
