# FP-P2D-022 Risk Note

Primary risk: shape-validation parity can drift under nested-list constructors, causing silent frame mis-shaping or nondeterministic diagnostics.

Mitigations:
1. Matrix packet covers square/rectangular/ragged inputs and multiple shape-error classes.
2. Differential harness enforces full-frame parity and critical classification for constructor mismatch/error drift.
3. Live oracle captures pandas behavior for both success and expected-error cases.

## Invariant Ledger Hooks

- `FP-I1` (shape consistency): constructor output shape is deterministic for accepted list-like inputs.
- `FP-I2` (missingness monotonicity): ragged rows produce deterministic null fills.
- `FP-I4` (determinism): repeated constructor payloads produce identical output/error outcomes.
- `FP-I7` (fail-closed semantics): malformed shape payloads reject explicitly.
