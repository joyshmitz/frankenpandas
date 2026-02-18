# FP-P2D-027 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-027` |
| input_contract | DataFrame `head_n` / `tail_n` signed selectors (`n` may be negative) |
| output_contract | deterministic row prefix/suffix materialization for both positive and negative `n` |
| error_contract | missing `frame` or missing signed selector fields fail closed |
| null_contract | null scalars are preserved exactly under negative-`n` slicing |
| index_contract | `head(-k)` preserves leading order; `tail(-k)` preserves trailing order after dropping first `k` |
| columns_contract | schema and column order preserved across head/tail materialization |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | bounded divergence only in explicitly allowlisted defensive categories |
| selector_scope | signed integer row-cardinality selectors for DataFrame head/tail |
| excluded_scope | axis=1 head/tail and MultiIndex-specific negative selector corner cases |
| oracle_tests | pandas `df.head(-k)` and `df.tail(-k)` parity baseline |
| performance_sentinels | bounded O(selected_rows * columns) copy path |
| compatibility_risks | off-by-one drop behavior for negative selectors |
| raptorq_artifacts | parity report + sidecar + decode proof emitted per packet run |
