# FP-P2D-027 Risk Note

Primary risk: negative-selector drift (`head(-k)` / `tail(-k)`) can silently drop wrong row ranges, causing downstream parity regressions in joins/groupby.

Mitigations:
1. Packet matrix covers strict+hardened negative selectors, saturation-to-empty behavior, and null-preservation.
2. Differential harness enforces fail-closed drift detection on index and value mismatches.
3. Signed selector normalization is centralized and reused for deterministic behavior.

## Invariant Ledger Hooks

- `FP-I1` (shape consistency): output cardinality follows signed selector contract.
- `FP-I4` (determinism): repeated signed selectors produce stable index/value order.
- `FP-I7` (fail-closed semantics): malformed fixtures reject explicitly.
