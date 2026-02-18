# FP-P2D-030 Legacy Anchor Map

Packet: `FP-P2D-030`  
Subsystem: DataFrame concat axis=0 `join='inner'` parity

## Pandas Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/reshape/concat.py` (`concat`, `_Concatenator`, axis=0 join selector behavior)
- `legacy_pandas_code/pandas/pandas/core/frame.py` (row-wise concat materialization and index handling)

## FrankenPandas Anchors

- `crates/fp-frame/src/lib.rs`: `concat_dataframes_with_axis_join`, `concat_dataframes_axis0_inner`
- `crates/fp-conformance/src/lib.rs`: packet execution and `concat_join` normalization
- `crates/fp-conformance/oracle/pandas_oracle.py`: pandas bridge for axis=0 inner selector
- `crates/fp-conformance/fixtures/packets/fp_p2d_030_*`: axis=0 inner fixture matrix

## Behavioral Commitments

1. `concat(axis=0, join='inner')` keeps only shared columns across both frames.
2. Output row index preserves left-then-right order with duplicates.
3. Existing null values in shared columns are preserved.

## Open Gaps

1. Axis=0 `join='outer'` union-column semantics are still fail-closed to exact-column match.
2. MultiIndex concat join semantics.
3. Cross-frame (>2 input) packet coverage for axis=0 inner.
