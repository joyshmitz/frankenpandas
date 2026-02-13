# FP-P2C-003 Risk Note

Primary risk: mixed-label and duplicate-label arithmetic semantics can drift from pandas in edge cases.

Mitigations:
1. strict mode remains fail-closed for unsupported compatibility surfaces.
2. hardened duplicate-label path is bounded and recorded.
3. packet gate thresholds enforce zero failed fixtures.
4. mismatch corpus is emitted each run for replay and drift triage.

## Isomorphism Proof Hook

- ordering preserved: yes for current union strategy (left order + right unseen append)
- tie-breaking preserved: yes for implemented first-hit behavior
- null/NaN/NaT behavior preserved: missing propagation on non-overlap paths is covered
- fixture checksum verification: tracked by `artifacts/perf/golden_checksums.txt`
