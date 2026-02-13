# FP-P2C-003 Legacy Anchor Map

Packet: `FP-P2C-003`
Subsystem: Series arithmetic + mixed-label alignment

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/series.py` (aligned binary arithmetic semantics)
- `legacy_pandas_code/pandas/pandas/core/indexes/base.py` (index union behavior and duplicate-label handling)

## Extracted Behavioral Contract

1. Alignment is label-driven and deterministic for union materialization.
2. Non-overlapping labels become missing values in arithmetic outputs.
3. Hardened duplicate-label path is explicit and auditable; strict mode remains fail-closed.

## Rust Slice Implemented

- `crates/fp-frame/src/lib.rs`: `Series::add_with_policy`
- `crates/fp-index/src/lib.rs`: `align_union`
- `crates/fp-conformance/src/lib.rs`: packetized differential fixture execution
