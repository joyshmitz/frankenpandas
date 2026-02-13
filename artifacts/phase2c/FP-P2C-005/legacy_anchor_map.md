# FP-P2C-005 Legacy Anchor Map

Packet: `FP-P2C-005`
Subsystem: groupby sum semantics

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/groupby/groupby.py` (groupby planning and reduction surfaces)
- `legacy_pandas_code/pandas/pandas/core/series.py` (series alignment and groupby entry points)

## Extracted Behavioral Contract

1. group key encounter order is preserved when sort is disabled.
2. missing keys are skipped under default `dropna=true` semantics.
3. missing values do not contribute to sums but keys remain materialized when encountered.

## Rust Slice Implemented

- `crates/fp-groupby/src/lib.rs`: `groupby_sum` with explicit alignment and first-seen ordering
- `crates/fp-conformance/src/lib.rs`: `groupby_sum` fixture operation and packet gate coverage
- `crates/fp-conformance/oracle/pandas_oracle.py`: live oracle adapter for `groupby_sum`
