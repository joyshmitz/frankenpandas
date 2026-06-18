# cod-a Negative-Evidence Ledger

Purpose: record every cod-a optimization attempt in the new performance campaign,
including benchmark-pending candidates and dead ends, so future agents do not
retry failed levers without a concrete retry predicate.

## 2026-06-18 - br-frankenpandas-8qn9i - Series.astype Column::astype route

- Status: implemented, benchmark verdict pending batch-test.
- Lever: route `Series::astype(dtype)` through `Column::astype(dtype)` instead
  of `Column::new(dtype, self.values().to_vec())`.
- Baseline comparator: current `Series::astype` materializes every value as a
  `Scalar`, then constructs a target-typed column through constructor coercion.
  This bypasses `Column::astype` typed Int64<->Float64 fast paths and also uses
  strict constructor semantics for finite non-integer Float64->Int64.
- Graveyard mapping: vectorized execution / typed data-plane specialization.
  The canonical graveyard guidance favors profile-backed, one-lever changes and
  data-structure specialization that avoids object materialization; the
  FrankenSuite summary calls out performance claims only after proof contracts
  and conservative fallback. This lever keeps the existing scalar fallback in
  `Column::astype` as the conservative path.
- Alien-artifact proof obligation: `Series::astype` should be a metadata
  wrapper around the column cast table. `Column::astype` already defines the
  pandas cast contract for Int64, Float64, Bool, Utf8, and missing values. The
  Series wrapper preserves series name, index labels, and index name while
  changing only the value dtype conversion engine.
- Guard added: `series_astype_uses_column_cast_table_8qn9i`, covering
  Float64->Int64 truncation toward zero, series name preservation, index label
  and index-name preservation, target dtype, and non-finite Float64->Int64
  rejection.
- Cass/ledger preflight: local `cass status --json` reported a stale lexical
  index; targeted `cass search "Series.astype Column::astype typed paths"`
  returned zero hits. Existing repo ledger search found no prior cod-a
  `Series.astype` attempt. The stale cass state is recorded here rather than
  treated as proof that no prior failed attempt exists.
- Validation run: passed
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenpandas-cod-a cargo check -p fp-frame`
  on 2026-06-18; only pre-existing workspace manifest license/license-file
  warnings were emitted.
- UBS run: `timeout 180s ubs crates/fp-frame/src/lib.rs` timed out with exit
  124 after entering the Rust scan, matching the known broad `fp-frame` scanner
  backlog/stall behavior documented in
  `artifacts/audits/fp_frame_ubs_inventory_2026-06-17.md`. No completed UBS
  finding was emitted for the touched hunk in that bounded window.
- Benchmark verdict: pending. Required follow-up comparator is a focused
  `Series.astype` Int64->Float64 and Float64->Int64 workload versus the legacy
  pandas original and a pre-patch scalar-materialization baseline, plus the
  existing astype conformance packet.
- Retry predicate if rejected: only revisit this wrapper route if same-host
  profiling shows `Series::astype` above 0.1% self-time and batch evidence
  proves the residual is scalar materialization or constructor coercion rather
  than unavoidable target column allocation.
