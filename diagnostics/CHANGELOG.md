# SREE Per-Block Diagnostics - Changelog

## [1.0.0] - 2024-12-01

### Added

- **Per-Block Diagnostics System**: Complete implementation of transparent, auditable logs per block
- **Feature Flagging**: `perBlockDiagnostics.enabled` flag (default: false) for backward compatibility
- **Data Models**:
  - `BlockRowDiagnostic`: Row-level diagnostic records
  - `BlockDiagnosticSummary`: Block-level aggregated summaries
  - `ColumnInsight`: Feature-level insights for logic validation
  - `DiagnosticsConfig`: Configuration management
- **Core Services**:
  - `DiagnosticsService`: Main service for managing diagnostics
  - `DiagnosticsHooks`: Integration hooks for existing code
  - `DiagnosticsAPI`: REST API endpoints
- **Integration Points**:
  - Automatic integration with `TrustUpdateLoop._make_row_decisions()`
  - Streamlit dashboard integration
  - Flask API routes
- **Storage & Export**:
  - In-memory storage with optional persistence
  - CSV export functionality
  - Configurable row limits to prevent memory explosion
- **Comprehensive Testing**: 29 test cases covering all functionality
- **Documentation**: Complete README with usage examples

### Features

- **Transparent Logging**: Track V_q (entropy), V_b (hash), V_l (logic) scores per row
- **Action Tracking**: Record DOWN_WEIGHTED, RETAINED, FLAGGED actions with reasons
- **Weight Deltas**: Track weight changes applied to rows
- **Column Insights**: Feature-level details when logic rules fail
- **Block Summaries**: Aggregated metrics per block for quick analysis
- **Performance Optimized**: Zero impact when disabled, sampling for large datasets
- **Audit Trail**: Timestamps and detailed explanations for auditors

### Configuration

```json
{
  "per_block_diagnostics": {
    "enabled": false, // Feature flag (default: false)
    "persist": false, // Persist to disk (default: false)
    "max_rows_per_block": 10000 // Max rows per block (default: 10000)
  }
}
```

### API Endpoints

- `GET /sree/diagnostics/:runId/blocks` → BlockDiagnosticSummary[]
- `GET /sree/diagnostics/:runId/block/:index` → BlockRowDiagnostic[]
- `GET /sree/diagnostics/:runId/export.csv` → CSV export
- `GET /sree/diagnostics/stats` → Statistics
- `DELETE /sree/diagnostics/:runId` → Clear run data

### Breaking Changes

- **None**: Fully backward compatible
- All existing functionality unchanged when feature flag is disabled

### Migration Guide

1. **No changes required**: System works with existing code
2. **To enable**: Set `perBlockDiagnostics.enabled = true` in configuration
3. **To disable**: Set `perBlockDiagnostics.enabled = false` (default)

### Testing

- ✅ Feature flag behavior (on/off)
- ✅ Data correctness and serialization
- ✅ API endpoints
- ✅ Integration hooks
- ✅ Edge cases and error handling
- ✅ Performance limits

### Performance Impact

- **When disabled**: Zero impact on performance or functionality
- **When enabled**: Minimal overhead, configurable limits prevent memory issues
- **Core metrics**: No change to accuracy, trust, or entropy calculations

### Security

- **No sensitive data exposure**: Only validation scores and actions
- **Configurable persistence**: Optional disk storage for auditing
- **Access control**: API endpoints can be protected with authentication

### Documentation

- Complete README with usage examples
- API documentation
- Integration guide
- Troubleshooting section
- Performance considerations

---

## Implementation Details

### Architecture

- **Modular Design**: Separate modules for types, service, hooks, and API
- **Global Service Pattern**: Easy access via `get_diagnostics_service()`
- **Lazy Loading**: Hooks only import diagnostics when needed
- **Error Handling**: Graceful degradation when diagnostics unavailable

### Integration Points

- **TrustUpdateLoop**: Automatic diagnostics in `_make_row_decisions()`
- **Dashboard**: New "Per-Block Diagnostics" section
- **Configuration**: Added to main config system
- **Logging**: Integrated with existing logging infrastructure

### Data Flow

1. **Row Processing**: Hooks emit diagnostics during validation
2. **Block Completion**: Summaries generated automatically
3. **Storage**: In-memory with optional persistence
4. **Access**: Via API or service methods
5. **Export**: CSV format for external analysis

### Future Enhancements

- **Real-time Dashboard**: Live updates during processing
- **Advanced Analytics**: Statistical analysis of diagnostics
- **Integration APIs**: Webhook support for external systems
- **Performance Monitoring**: Metrics on diagnostic overhead
