# FP-P2D-030 Risk Note

Primary risk: axis=0 inner-join concat can silently drift on shared-column selection or index ordering, producing downstream schema or alignment regressions.

Mitigations in this packet:

1. Overlap/disjoint fixture matrix locks shared-column intersection semantics.
2. Null and empty-schema cases validate robustness under sparse inputs.
3. Error fixtures enforce fail-closed selector validation.

Invariant hooks:

- `FP-I1` (shape consistency): output row count equals `len(left) + len(right)`.
- `FP-I4` (determinism): shared-column selection is stable for repeated runs.
- `FP-I7` (fail-closed semantics): invalid selector inputs return explicit errors.

Residual risk:

- Axis=0 outer union-column parity remains intentionally out of scope for this packet.
- MultiIndex axis=0 join semantics remain unimplemented.
