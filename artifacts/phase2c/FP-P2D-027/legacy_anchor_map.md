# FP-P2D-027 Legacy Anchor Map

Packet: `FP-P2D-027`
Subsystem: DataFrame `head`/`tail` negative-`n` semantics

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/generic.py` (`NDFrame.head`/`NDFrame.tail` signed selector behavior)
- `legacy_pandas_code/pandas/pandas/core/frame.py` (DataFrame row slicing and index preservation)

## Extracted Behavioral Contract

1. `df.head(-k)` returns all rows except the last `k`.
2. `df.tail(-k)` returns all rows except the first `k`.
3. If `k >= len(df)`, result is an empty-row frame with unchanged column schema.
4. Scalar/null payloads are copied without coercion.

## Rust Slice Implemented

- `crates/fp-frame/src/lib.rs`: signed `DataFrame::head`/`tail` selector normalization.
- `crates/fp-conformance/src/lib.rs`: signed fixture payload handling for `head_n`/`tail_n`.
- `crates/fp-conformance/fixtures/packets/fp_p2d_027_*`: negative-selector packet matrix.

## Rule Ledger

1. Negative selector semantics are deterministic and saturating.
2. Column schema remains unchanged for empty outputs.
3. Index ordering remains stable under signed slicing.

## Undefined-Behavior Edges

1. Signed selector overflow beyond 64-bit fixture bounds.
2. Axis=1 signed head/tail semantics.
