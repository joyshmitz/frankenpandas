# ROUND4 Recommendation Contract

Change:
- Add dense `Int64` aggregation path to `groupby_sum` with range budget fallback.

Hotspot evidence:
- Flamegraph before showed hash-dominant runtime in groupby key accumulation and index-label hashing.

Mapped graveyard sections:
- `§7.7` Swiss-table/hash-probe pressure (hash-heavy symptom class)
- composition guidance: bounded fast path + deterministic fallback

EV score:
- `(Impact 3 * Confidence 4 * Reuse 4) / (Effort 4 * Friction 1) = 12.0`

Priority tier:
- `A`

Adoption wedge:
- Limited to `groupby_sum` internals; no API contract changes.

Budgeted mode:
- Dense path only when key span `<= 65_536`; otherwise fallback to generic path.

Expected-loss model:
- States: `{dense_eligible, dense_ineligible}`
- Actions: `{dense_path, generic_path}`
- Loss: prefer dense for eligible keys (lower latency), generic for ineligible keys (bounded memory/complexity)

Calibration + fallback trigger:
- Fallback is immediate if key domain violates `Int64`/span constraints.

Isomorphism proof plan:
- Unit tests on ordering, duplicate-index behavior, null-group fallback, and golden checksum verification.

p50/p95/p99 target:
- Improve round4 baseline while preserving strict/hardened parity.

Primary failure risk + countermeasure:
- Risk: semantic drift for null/mixed keys.
- Countermeasure: strict eligibility gate + generic-path fallback + targeted tests.

Rollback:
- Revert dense-path helper and call site in `fp-groupby`.

Baseline comparator:
- Round4 pre-change benchmark (`round4_groupby_hyperfine_before.json`).
