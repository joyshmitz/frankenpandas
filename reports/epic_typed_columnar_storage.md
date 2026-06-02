FLAGSHIP vs-upstream gap (orchestrator campaign: no perf gap is acceptable; build our own safe-Rust typed columnar, NOT link C BLAS/numpy). Supersedes/expands piw16.

## Baseline (RubyGoose, release-perf, 2026-06-02, 100k rows x10 Float64)
- filter_bool:     FP 4.868 ms  vs pandas 0.423 ms  = 11.5x   <-- biggest gap
- sort_single:     FP 12.190 ms vs pandas 5.563 ms  = 2.2x
- drop_duplicates: FP 19.506 ms vs pandas 14.941 ms = 1.3x

## Root cause (single, shared)
Column stores `values: Vec<Scalar>` (AoS); Scalar is a 32-byte enum (Utf8(String)/Interval are the wide variants). filter/sort materialize the result by gathering+cloning 32-byte Scalars and rebuilding a Vec<Scalar> output. filter_bool gather = 71% self in Column::take_positions; sort_single routes through the SAME take_rows_by_positions gather (1M x 32B = 32MB/iter). numpy gathers contiguous 8-byte f64.

Confirmed NO contained lever: any typed gather that still emits Vec<Scalar> is net-NEGATIVE (adds an O(n) extraction pass); the output rebuild is the wall. The orchestration layer (take_rows_by_positions / take_positions) is already fully fast-pathed (sfysu / 2a6ln / 8cc83365) -- nothing left at that layer.

## Alien primitives (alien_cs_graveyard.md)
- Struct-of-Arrays contiguous columnar layout.
- SIMD/branchless gather via selection vectors (positions -> std::simd gather on typed buffers, no per-element enum match).
- Section 7.2 cache-oblivious / cache-blocked access for frames larger than LLC.
- forbid(unsafe_code) stays: std::simd / autovectorizable loops only.

## Target architecture
`Column { dtype, data: ColumnData (typed Vec<f64>/Vec<i64>/...), validity: ValidityMask, scalars: OnceCell<Vec<Scalar>> }`. ColumnData already EXISTS (fp-columnar:328) but is only materialized on demand -- make it the source of truth. Hot kernels (take_positions, filter, sort gather, compare) operate on typed data and emit typed-backed Columns with EMPTY scalar cache -> zero Scalar materialization on hot paths. `values() -> &[Scalar]` lazily materializes via OnceCell for the ~5400 legacy callers (compat preserved).

## Blast radius / feasibility
Column.values field is PRIVATE; only 9 direct `Column {` struct literals repo-wide (5 fp-columnar, 3 fp-frame, 1 fp-conformance) + constructors new/from_values. But 334 internal self.values[..] reads in fp-columnar must move to a private typed accessor or typed kernels. All reads (no values_mut / no mutation aliasing) -> lazy materialization is sound.

## Isomorphism hazards (must prove; ABSOLUTE parity)
1. NULL-KIND PRESERVATION -- values() currently returns the exact Scalar::Null(NaN|NaT|Null) per invalid position; typed ColumnData+validity loses null-kind. Reconstruct via dtype canonicalization: take_positions already calls normalize_missing_for_dtype, so nulls are dtype-canonical (Float64->NaN, Datetime64->NaT, Int64->Null, ...). MUST verify this invariant holds at EVERY constructor before relying on it; otherwise store a per-column null-kind (or per-position kind).
2. Float bit pattern: gather copies bits, no arithmetic -> identical.
3. Ordering/tie-break: gather preserves positional order -> identical.
4. Utf8/Interval columns: keep Vec<String>/Vec<Interval> typed; gather clones same as before.

## Phasing (one lever per commit; re-bench each; Score>=2.0 to keep)
- P0 (groundwork): baseline + plan (DONE -- this bead).
- P1: introduce typed-backed Column + OnceCell values() with FULL parity (all crates green + conformance + golden sha256 of filter/sort/gather outputs vs current). No behavior change; sets storage.
- P2: typed std::simd gather in take_positions for Float64/Int64/Bool -> realizes filter_bool + sort_single win. Re-bench; require >=2x.
- P3: extend typed kernels to compare / sort-key / groupby-hash; re-profile.

## Proof protocol per phase
rch crate-scoped build/test; full workspace cargo test; fp-conformance differential vs oracle; golden sha256 of operation outputs captured pre-change must match post-change; hyperfine / perf_profile before+after numbers in commit. Prove isomorphism or revert. Push main + main:master.
