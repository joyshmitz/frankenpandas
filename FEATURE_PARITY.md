# FEATURE_PARITY

## Status Legend

- not_started
- in_progress
- parity_green
- parity_gap

## Parity Matrix

| Feature Family | Status | Notes |
|---|---|---|
| DataFrame/Series constructors | in_progress | `Series::from_values` + `DataFrame::from_series` MVP implemented; `FP-P2C-003` extends arithmetic fixture coverage; broader constructor parity pending |
| Index alignment and selection | in_progress | `FP-P2C-001`/`FP-P2C-002` packet suites green with gate validation and RaptorQ sidecars; `loc/iloc` parity still pending |
| GroupBy core aggregates | in_progress | `FP-P2C-005` packet suite green for first-seen key ordering + dropna alignment semantics; broader aggregate matrix still pending |
| Join/merge/concat core | in_progress | `FP-P2C-004` packet suite green for series-level `inner`/`left` semantics; full DataFrame merge/concat contracts pending |
| Null/NaN semantics | in_progress | missing propagation in arithmetic + scalar semantic equality implemented; nanops matrix pending |
| Core CSV ingest/export | in_progress | first CSV read/write path implemented in `fp-io`; parser/formatter parity matrix pending |

## Phase-2C Packet Evidence (Current)

| Packet | Result | Evidence |
|---|---|---|
| FP-P2C-001 | parity_green | `artifacts/phase2c/FP-P2C-001/parity_report.json`, `artifacts/phase2c/FP-P2C-001/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-001/parity_report.raptorq.json` |
| FP-P2C-002 | parity_green | `artifacts/phase2c/FP-P2C-002/parity_report.json`, `artifacts/phase2c/FP-P2C-002/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-002/parity_report.raptorq.json` |
| FP-P2C-003 | parity_green | `artifacts/phase2c/FP-P2C-003/parity_report.json`, `artifacts/phase2c/FP-P2C-003/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-003/parity_report.raptorq.json` |
| FP-P2C-004 | parity_green | `artifacts/phase2c/FP-P2C-004/parity_report.json`, `artifacts/phase2c/FP-P2C-004/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-004/parity_report.raptorq.json` |
| FP-P2C-005 | parity_green | `artifacts/phase2c/FP-P2C-005/parity_report.json`, `artifacts/phase2c/FP-P2C-005/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-005/parity_report.raptorq.json` |

Gate enforcement and trend history:

- blocking command: `./scripts/phase2c_gate_check.sh`
- CI workflow: `.github/workflows/ci.yml`
- drift history ledger: `artifacts/phase2c/drift_history.jsonl`

## Required Evidence Per Feature Family

1. Differential fixture report.
2. Edge-case/adversarial test results.
3. Benchmark delta (when performance-sensitive).
4. Documented compatibility exceptions (if any).
