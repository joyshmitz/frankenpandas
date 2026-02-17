# FP-P2D-023 Risk Note

Primary risk: constructor dtype/copy pathways can drift silently, yielding type coercion mismatches or unstable error taxonomies that break pandas-observable behavior.

Mitigations:
1. Packet covers supported dtype coercions across constructor entry points (`from_series`, `from_dict`, `from_records`, kwargs, scalar, dict-of-series, list-like).
2. Error fixtures lock deterministic diagnostics for unsupported dtype specs and lossy/invalid casts.
3. Differential harness enforces strict gate failure on constructor parity regressions.

## Invariant Ledger Hooks

- `FP-I1` (shape consistency): constructor output shape remains stable after dtype option application.
- `FP-I2` (missingness monotonicity): dtype coercion remaps missing values deterministically.
- `FP-I4` (determinism): repeated constructor payloads and options produce stable output/error outcomes.
- `FP-I7` (fail-closed semantics): invalid dtype specs and non-castable values reject explicitly.
