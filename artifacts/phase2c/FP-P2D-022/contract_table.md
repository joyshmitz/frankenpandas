# FP-P2D-022 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-022` |
| input_contract | nested-list / 2D `matrix_rows` constructor payloads with optional `index` and `column_order` |
| output_contract | deterministic shape realization for square/rectangular/ragged row matrices |
| error_contract | shape mismatches fail closed with explicit diagnostics (`row width` overflow, `index length` mismatch, missing payload) |
| null_contract | ragged and expanded-column surfaces deterministically null-fill missing cells |
| index_contract | explicit index must match row cardinality; default range index otherwise |
| columns_contract | explicit columns constrain/expand shape; default positional string columns when omitted |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | bounded divergence only in explicitly allowlisted defensive categories |
| constructor_scope | shape taxonomy coverage: square, rectangular, ragged, empty rows, empty columns, mixed scalar domains |
| excluded_scope | non-scalar nested payloads, MultiIndex constructor paths, extension-array constructor semantics |
| oracle_tests | pandas `DataFrame(matrix_rows, index=..., columns=...)` parity baseline |
| performance_sentinels | constructor cost under ragged-row and explicit-shape validation paths |
| compatibility_risks | shape-validation drift, default-column-label drift, null-kind drift in ragged matrices |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
