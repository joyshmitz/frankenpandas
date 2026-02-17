# FP-P2D-024 Risk Note

Primary risk: dtype-spec parsing drift can silently change constructor coercion behavior or destabilize diagnostic surfaces for unsupported types.

Mitigations:
1. Packet anchors alias normalization (`INT64`, `Float`, `BOOLEAN`, `str`, `STRING`, `f64`) across constructor paths.
2. Unsupported dtype taxonomy fixtures lock deterministic fail-closed diagnostics.
3. Differential harness enforces packet gate failure on normalization/error drift.

## Invariant Ledger Hooks

- `FP-I1` (shape consistency): dtype normalization does not alter constructor shape semantics.
- `FP-I4` (determinism): alias inputs produce stable output and unsupported specs produce stable diagnostics.
- `FP-I7` (fail-closed semantics): unsupported dtype specs reject explicitly.
