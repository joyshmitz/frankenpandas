# FP-P2C-004 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-004` |
| input_contract | two indexed series (`left`, `right`) with explicit join type (`inner`/`left`) |
| output_contract | joined index plus paired left/right value columns |
| error_contract | unknown join mode fails closed |
| null_contract | left join emits missing right values for unmatched labels |
| index_alignment_contract | duplicate keys multiply cardinality according to join mode |
| strict_mode_policy | fail-closed on unknown incompatible semantics |
| hardened_mode_policy | bounded duplicate-label continuation with evidence logging hooks |
| excluded_scope | full DataFrame multi-column merge API surface and sort semantics matrix |
| oracle_tests | `tests/reshape/merge/test_merge.py` mapped subset |
| performance_sentinels | duplicate-key join cardinality expansion and allocation profile |
| compatibility_risks | duplicate-key ordering and missing marker normalization drift |
| raptorq_artifacts | parity report, sidecar, gate result, mismatch corpus, decode proof |
