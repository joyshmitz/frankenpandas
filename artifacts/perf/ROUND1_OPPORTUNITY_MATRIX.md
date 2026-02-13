# Round 1 Opportunity Matrix

| Hotspot | Impact (1-5) | Confidence (1-5) | Effort (1-5) | Score | Decision |
|---|---:|---:|---:|---:|---|
| `fp-conformance::run_fixture` duplicate operation call | 3 | 5 | 1 | 15.0 | implement |
| Python oracle process spawn / `execve` dominance in packet pipeline | 4 | 4 | 3 | 5.3 | queue for next round |
| Repeated JSON parse per fixture | 4 | 3 | 3 | 4.0 | queue for next round |
| CSV parser fallback coercion path | 2 | 3 | 3 | 2.0 | queue for next round |

Applied lever this round:

- removed duplicate `run_fixture_operation` invocation inside `run_fixture` and kept single execution with captured mismatch.

Fallback path:

- if any mismatch appears after optimization, revert to explicit two-phase validation in `run_fixture` and compare behavior against `golden_checksums.txt`.
