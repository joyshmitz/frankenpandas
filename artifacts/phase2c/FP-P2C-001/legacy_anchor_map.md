# FP-P2C-001 Legacy Anchor Map

Packet: `FP-P2C-001`
Subsystem: DataFrame/Series construction + alignment

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/frame.py` (`DataFrame`, `_from_nested_dict`, `_reindex_for_setitem`)
- `legacy_pandas_code/pandas/pandas/core/series.py` (`Series` construction and aligned binary arithmetic)

## Extracted Behavioral Contract

1. Index alignment is label-driven and must materialize a deterministic union before arithmetic.
2. Missing labels introduce missing values, not dropped rows.
3. Duplicate index handling is compatibility-critical and must be mode-gated.

## Rust Slice Implemented

- `crates/fp-frame/src/lib.rs`: `Series::from_values`, `Series::add_with_policy`, `DataFrame::from_series`
- `crates/fp-index/src/lib.rs`: deterministic union alignment plan (`align_union`)
- `crates/fp-columnar/src/lib.rs`: reindexing and missing propagation in numeric ops
