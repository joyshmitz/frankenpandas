# FP-P2C-004 Legacy Anchor Map

Packet: `FP-P2C-004`
Subsystem: indexed join semantics

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/reshape/merge.py` (merge/join planning)
- `legacy_pandas_code/pandas/pandas/core/series.py` (index-driven join behavior for aligned operations)

## Extracted Behavioral Contract

1. inner joins on duplicate keys expand to cross-product cardinality.
2. left joins preserve left ordering and inject missing right values for unmatched keys.
3. hardened mode is allowed bounded continuation but strict mode remains fail-closed on unknown surfaces.

## Rust Slice Implemented

- `crates/fp-join/src/lib.rs`: `join_series` for `Inner` and `Left`
- `crates/fp-conformance/src/lib.rs`: `series_join` fixture operation and packet gate coverage
