# FP-P2C-001 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-001` |
| input_contract | Two series with explicit index labels and scalar payloads |
| output_contract | Aligned index union with arithmetic result values |
| error_contract | strict mode rejects unsupported duplicate-label behavior |
| null_contract | missing on either side propagates to result (`Null` or `NaN`) |
| index_alignment_contract | left-order-preserving union with right-unseen append |
| strict_mode_policy | fail-closed on unsupported duplicate-label semantics |
| hardened_mode_policy | allow bounded repair path; decision logged to evidence ledger |
| excluded_scope | full pandas duplicate label semantics, `loc`/`iloc`, broadcast rules |
| oracle_tests | `tests/frame/test_constructors.py`, `tests/series/test_arithmetic.py` (mapped subset) |
| performance_sentinels | 10M-row alignment arithmetic p95 + allocation churn |
| compatibility_risks | silent duplicate-label drift; null marker mismatch |
| raptorq_artifacts | packet parity report + fixture manifest sidecars |
