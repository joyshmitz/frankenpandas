# FP-P2D-024 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-024` |
| input_contract | constructor payloads with normalized dtype spec strings (case, whitespace, aliases) |
| output_contract | supported aliases map deterministically to internal dtype coercion |
| error_contract | unsupported dtype spec strings fail closed with explicit unsupported-dtype diagnostics |
| null_contract | supported dtype normalization preserves existing missingness rules |
| index_contract | constructor index behavior is unchanged by dtype-spec normalization |
| columns_contract | column projection/order behavior is unchanged by dtype-spec normalization |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | bounded divergence only in explicitly allowlisted defensive categories |
| constructor_scope | dtype string normalization and unsupported spec taxonomy |
| excluded_scope | extension-array backend semantics beyond explicit unsupported-spec rejection |
| oracle_tests | pandas constructor dtype argument handling baseline for covered alias/error shapes |
| performance_sentinels | negligible overhead from dtype-spec normalization and unsupported-spec checks |
| compatibility_risks | alias-normalization drift and unsupported-spec message drift |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
