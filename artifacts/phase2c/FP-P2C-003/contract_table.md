# FP-P2C-003 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-003` |
| input_contract | Series arithmetic inputs across `utf8` and `int64` index labels |
| output_contract | deterministic union ordering and expected value materialization |
| error_contract | strict mode rejects unsupported incompatibilities; hardened mode logs bounded repairs |
| null_contract | non-overlapping labels produce missing markers in result |
| index_alignment_contract | left-order-preserving union with right-unseen append |
| strict_mode_policy | fail-closed for incompatible/unknown behavior surfaces |
| hardened_mode_policy | duplicate-label repair permitted with ledger visibility |
| excluded_scope | full pandas duplicate-label semantics and advanced broadcast rules |
| oracle_tests | `tests/series/test_arithmetic.py` mapped subset |
| performance_sentinels | utf8-label alignment throughput and allocation behavior |
| compatibility_risks | duplicate-label and mixed-label drift vs pandas edge semantics |
| raptorq_artifacts | packet parity report + sidecar + decode proof + mismatch corpus |
