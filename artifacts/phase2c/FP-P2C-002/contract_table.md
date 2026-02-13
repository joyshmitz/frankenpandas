# FP-P2C-002 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-002` |
| input_contract | index label vectors (`int64` and `utf8`) |
| output_contract | deterministic alignment plan with positional maps |
| error_contract | invalid alignment vectors fail explicitly |
| null_contract | not applicable at index-only layer |
| index_alignment_contract | first-occurrence position map with deterministic union |
| strict_mode_policy | duplicate-label unsupported path is rejected upstream |
| hardened_mode_policy | duplicate-label path can continue under logged repair |
| excluded_scope | `MultiIndex`, partial-string slicing, timezone index semantics |
| oracle_tests | `tests/indexes/test_base.py`, `tests/indexing/*` |
| performance_sentinels | index union throughput + hash-map contention |
| compatibility_risks | first-occurrence mapping differs from full pandas duplicate semantics |
| raptorq_artifacts | index packet reports and fixture sidecars |
