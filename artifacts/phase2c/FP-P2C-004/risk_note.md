# FP-P2C-004 Risk Note

Primary risk: duplicate-key join cardinality and missing-right marker behavior can drift from pandas-observable semantics.

Mitigations:
1. strict packet gate enforces zero failed fixtures.
2. hardened duplicate path remains explicit and audited.
3. mismatch corpus is emitted every run for replay.
4. drift history records packet-level pass/fail trends.

## Isomorphism Proof Hook

- ordering preserved: left-driven output ordering is deterministic in current implementation
- tie-breaking preserved: duplicate-key expansion follows stable nested loop order
- null/NaN/NaT behavior preserved: unmatched right rows map to missing scalar markers
- fixture checksum verification: tracked in `artifacts/perf/golden_checksums.txt`
