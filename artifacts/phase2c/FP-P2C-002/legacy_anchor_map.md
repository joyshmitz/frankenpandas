# FP-P2C-002 Legacy Anchor Map

Packet: `FP-P2C-002`
Subsystem: Index model and indexer semantics

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/indexes/base.py` (`Index`, `ensure_index`, `_validate_join_method`)

## Rust Slice Implemented

- `crates/fp-index/src/lib.rs`: `IndexLabel`, `Index`, `align_union`, duplicate detection
- `crates/fp-frame/src/lib.rs`: strict/hardened duplicate-index compatibility gate

## Deferred

- full `get_indexer` method matrix
- `MultiIndex` tuple semantics
- full error-string parity against pandas
