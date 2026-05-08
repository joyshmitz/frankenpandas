# Null and NaN Metamorphic Matrix - 2026-05-08

Bead: `br-frankenpandas-tn6qb.6`

Scope: strict-mode pandas-observable behavior nets for null and NaN propagation.
Hardened-mode recovery behavior is intentionally out of scope for these tests
unless explicitly called out in a follow-up bead.

## Property Matrix

| MR | Subsystem | Transformation | Invariant | Covers | Score |
| --- | --- | --- | --- | --- | --- |
| MR-01 | Series reindex/dropna | Reindex to a target containing duplicated missing labels, then `dropna()` | Result equals `dropna()` on the original Series; signed zero payload is preserved | Null, NaN, signed zero, missing labels, duplicate target labels | 4.5 |
| MR-02 | DataFrame concat/alignment | `concat(axis=1, join=outer)` over misaligned frames | Output index and each side's materialized columns match direct outer `align_on_index()` | NaN fills, missing labels, index alignment, cross-column dtype promotion | 4.2 |
| MR-03 | SeriesGroupBy | Insert rows whose group key is Null/NaN | Non-missing groups keep the same aggregates; the missing-key bucket is isolated | Null keys, NaN keys, missing bucket coalescing, groupby aggregation | 4.0 |
| MR-04 | Join/merge | Join with Null/NaN key permutations | Strict-mode key matching/nonmatching matches pandas; hardened recovery remains separate | Null keys, NaN keys, duplicate labels, mixed dtype keys | Follow-up `br-frankenpandas-tn6qb.8` |
| MR-05 | Reshape | Pivot/stack/melt under missing labels and Null/NaN values | Shape-preserving transforms do not invent or drop non-missing observations | Null, NaN, duplicate labels, signed-zero labels where observable | Follow-up `br-frankenpandas-tn6qb.9` |

## Implemented Tests

- `null_nan_metamorphic_series_reindex_dropna_identity_tn6qb6`
- `null_nan_metamorphic_concat_axis1_matches_outer_alignment_tn6qb6`
- `null_nan_metamorphic_series_groupby_missing_key_isolation_tn6qb6`

## Mode Boundary

These tests assert strict pandas parity only. Any hardened-mode recovery guard
must use distinct test names, distinct fixtures, and explicit policy setup so it
cannot silently change strict-mode null or NaN semantics.

## Validation Evidence

- `rch exec -- cargo test -p fp-frame null_nan_metamorphic -- --nocapture`: 3 passed, 0 failed.
- `rch exec -- cargo check -p fp-frame --all-targets`: passed.
- `cargo fmt --check`: passed.
- `ubs crates/fp-frame/src/lib.rs artifacts/null-nan-metamorphic-matrix-2026-05-08.md .beads/issues.jsonl`: timed out after 180 seconds on the large `fp-frame` Rust scan with no findings emitted.
- Local added-block hazard scan for `unwrap`, `expect`, `panic`, `todo`, `unimplemented`, and `unsafe`: no matches in the three new metamorphic tests.
