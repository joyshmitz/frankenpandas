# FP-P2C-005 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-005` |
| input_contract | two indexed series (`left=keys`, `right=values`) |
| output_contract | aggregate `sum` series indexed by first-seen key order |
| error_contract | incompatible payload shapes fail closed |
| null_contract | `dropna=true` skips missing keys and treats missing values as additive no-op |
| index_alignment_contract | union alignment preserves left order plus right-only labels; key/value alignment is explicit |
| strict_mode_policy | zero-drift parity gate with fail-closed unknown semantic surfaces |
| hardened_mode_policy | bounded continuation with explicit divergence allowlist and evidence hooks |
| excluded_scope | multi-aggregate matrix (`mean`, `count`, `min/max`) and multi-key DataFrame groupby |
| oracle_tests | pandas `groupby(...).sum()` contract slice via oracle adapter |
| performance_sentinels | key cardinality skew and alignment-induced sparse rows |
| compatibility_risks | first-seen ordering drift, NaN/null key handling mismatch |
| raptorq_artifacts | parity report, sidecar, gate result, mismatch corpus, decode proof |
