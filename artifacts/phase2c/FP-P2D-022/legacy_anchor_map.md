# FP-P2D-022 Legacy Anchor Map

Packet: `FP-P2D-022`
Subsystem: DataFrame list-like constructor shape/error taxonomy parity

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/frame.py` (`DataFrame.__init__` list-like constructor path)
- `legacy_pandas_code/pandas/pandas/core/internals/construction.py` (shape coercion and mismatch diagnostics)
- `legacy_pandas_code/pandas/pandas/core/indexes/base.py` (index cardinality semantics)

## Extracted Behavioral Contract

1. List-like matrices map rows to index cardinality and columns to row width (or explicit columns).
2. Explicit columns narrower than row width are invalid.
3. Explicit index length must equal row count.
4. Ragged rows null-fill missing cells deterministically when shape contract allows.
5. Missing required list-like payload fails closed.

## Rust Slice Implemented

- `crates/fp-conformance/src/lib.rs`: list-like constructor execution and validation diagnostics
- `crates/fp-conformance/oracle/pandas_oracle.py`: pandas-backed list-like constructor oracle path
- `crates/fp-frame/src/lib.rs`: materialization primitives used by constructor replay

## Type Inventory

- `fp_conformance::FixtureOperation::DataFrameConstructorListLike`
- `fp_conformance::PacketFixture` fields: `matrix_rows`, `index`, `column_order`, `expected_frame`, `expected_error_contains`
- `fp_types::Scalar`: `Int64`, `Float64`, `Utf8`, `Null`

## Rule Ledger

1. Shape validation errors are classified as hard constructor failures.
2. Default columns are positional string labels.
3. Null fill semantics are deterministic under ragged rows.
4. Error fixtures pass only when failure text contains expected substrings.

## Hidden Assumptions

1. Matrix payload rows contain scalar values only.
2. Current Rust constructor path emits positional columns as UTF-8 string labels.

## Undefined-Behavior Edges

1. Duplicate explicit column labels.
2. Nested non-scalar object payloads in matrix rows.
