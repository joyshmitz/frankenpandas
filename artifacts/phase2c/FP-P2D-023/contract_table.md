# FP-P2D-023 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-023` |
| input_contract | constructor payloads across list-like, dict/records, scalar, kwargs, and from-series paths with optional `constructor_dtype` and `constructor_copy` |
| output_contract | deterministic constructor materialization after optional dtype coercion |
| error_contract | unsupported dtype specs and invalid casts fail closed with explicit diagnostics |
| null_contract | dtype coercion preserves missingness markers using dtype-appropriate null semantics |
| index_contract | constructor index behavior remains operation-scoped (default range index or explicit index contract) |
| columns_contract | column projection/order semantics are unchanged by dtype/copy options |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | bounded divergence only in explicitly allowlisted defensive categories |
| constructor_scope | dtype coercion semantics (`bool`, `int64`, `float64`, `utf8`) and `copy` option acceptance |
| excluded_scope | pandas extension-array backends, object dtype fallback semantics, and copy-on-write mutation observation |
| oracle_tests | pandas constructor parity baseline for dtype/copy argument handling |
| performance_sentinels | constructor throughput under per-column dtype cast paths |
| compatibility_risks | dtype-cast drift, cast-error taxonomy drift, missingness remapping drift |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
