# FP-P2C-011 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-011` |
| input_contract | indexed value series + either single key series (`left`) or multi-key key series list (`groupby_keys`) with aggregate function specification |
| output_contract | aggregate series indexed by first-seen group key order; multi-key fixtures encode tuple keys deterministically as composite utf8 labels |
| error_contract | incompatible payload shapes, invalid alignment plans fail closed |
| null_contract | null keys excluded when dropna=true; null values skipped in aggregations; all-null group semantics per aggregate function |
| index_alignment_contract | keys/values aligned via union; output preserves first-seen key order |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | divergence only in explicit allowlisted defensive classes |
| excluded_scope | full DataFrameGroupBy API surface (`as_index`, named aggregation, transform/filter/apply), categorical observed parameter, rolling/expanding |
| oracle_tests | pandas `groupby().mean/count/min/max/first/last/std/var/median` via oracle adapter, including multi-key composite-key fixtures (strict + hardened) |
| performance_sentinels | group cardinality skew, dense-int path eligibility, arena budget overflow |
| compatibility_risks | aggregate empty/all-null group semantics, first/last null-skipping semantics, std/var ddof handling, multi-key null-component drop semantics, first-seen order stability |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
