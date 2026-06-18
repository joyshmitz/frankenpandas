# cod-b Negative-Evidence Ledger

Purpose: record every cod-b optimization attempt in the new performance campaign,
including dead ends, so future agents do not retry failed levers without a concrete
retry predicate.

## 2026-06-18 - br-frankenpandas-uza04.188 - MultiIndex tuple set-op FxHashSet

- Status: implemented, benchmark verdict pending batch-test.
- Lever: replace private `HashMap<Vec<IndexLabel>, ()>` SipHash membership/seen
  maps in `MultiIndex::{intersection, union, difference, symmetric_difference}`
  generic tuple fallback paths with `FxHashSet<Vec<IndexLabel>>`.
- Baseline comparator: current std `HashMap` SipHash tuple-key fallback.
- Graveyard mapping: `alien_cs_graveyard.md` section 7.7 Swiss Tables / high
  performance hash maps, plus the suite summary quick-fix guidance to replace
  default SipHash on non-DoS-facing internal maps.
- Alien-artifact proof obligation: output order remains input-scan driven, not
  hash-iteration driven; tuple membership identity is unchanged; no public
  `HashMap` return type changes.
- Guard added: `multi_index_setop_generic_fallback_preserves_order_codb`,
  forcing the non-packed fallback with 65 levels and checking intersection,
  union, difference, and symmetric difference ordering/dedup semantics.
- Validation run: passed `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenpandas-cod-b cargo check -p fp-index`
  on 2026-06-18; only pre-existing workspace manifest license/license-file
  warnings were emitted. Re-ran after the shared checkout advanced to
  `2edf0cf7` with the same pass result.
- Benchmark verdict: pending. Required follow-up comparator is a focused
  MultiIndex set-op workload where mixed-radix packed keys decline and tuple-key
  fallback dominates.
- Retry predicate if rejected: only retry if a profile shows this exact generic
  fallback above 0.1% self-time and a same-host benchmark with this patch reverted
  proves SipHash probe cost, not tuple construction, is the dominant residual.
