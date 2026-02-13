# FP-P2C-002 Risk Note

Primary risk: duplicate-label and indexer edge-case behavior drift from pandas.

Mitigations:
1. keep strict mode fail-closed for unsupported duplicate-label cases.
2. mark hardened repairs with explicit evidence ledger records.
3. dedicated `FP-P2C-002` fixture family covers align-union, duplicate detection, and first-position maps.
4. packet gate (`parity_gate.yaml`) is evaluated with strict/hardened counters and drift budgets.
5. mismatch corpus is emitted on every artifact write for differential replay.

## Isomorphism Proof Hook

- ordering preserved: yes for implemented union path
- tie-breaking preserved: first occurrence wins; full pandas tie behavior deferred
- null/NaN/NaT behavior preserved: not applicable at index-only layer
- fixture checksum verification: complete (`artifacts/perf/golden_checksums.txt`)
