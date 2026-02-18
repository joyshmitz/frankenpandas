# FP-P2D-030 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-030` |
| input_contract | two DataFrame payloads (`frame`, `frame_right`) with `concat_axis=0` and `concat_join=inner` |
| output_contract | `concat(axis=0, join='inner')` materializes row-wise concat over shared column intersection |
| error_contract | invalid `concat_axis` and invalid `concat_join` fail closed |
| null_contract | existing null values are preserved in intersected columns |
| index_alignment_contract | output index is left labels followed by right labels with duplicates preserved |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | divergence only in explicit allowlisted defensive classes |
| selector_scope | `dataframe_concat` with `concat_axis=0` and `concat_join='inner'` |
| excluded_scope | axis=0 `join='outer'` union-column semantics, MultiIndex concat, >2-frame packet matrices |
| oracle_tests | pandas `concat(axis=0, join='inner', sort=False)` over overlap/disjoint/null/empty/error matrices |
| performance_sentinels | intersection-column filtering cost, empty-intersection path, duplicate-index preservation |
| compatibility_risks | column-intersection drift, index concatenation drift, selector normalization drift |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
