# FP-P2D-024 Legacy Anchor Map

Packet: `FP-P2D-024`
Subsystem: DataFrame constructor dtype-spec normalization/error taxonomy

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/frame.py` (`DataFrame.__init__` dtype parsing)
- `legacy_pandas_code/pandas/pandas/core/dtypes/common.py` (dtype parsing aliases)
- `legacy_pandas_code/pandas/pandas/core/dtypes/cast.py` (unsupported dtype behavior)

## Extracted Behavioral Contract

1. Dtype strings are normalized across case and surrounding whitespace.
2. Known aliases map deterministically to constructor coercion behavior.
3. Unsupported dtype strings fail explicitly and deterministically.
4. Constructor copy/index/column behavior remains stable under dtype normalization.

## Rust Slice Implemented

- `crates/fp-conformance/src/lib.rs`: dtype-spec parsing and constructor option application
- `crates/fp-conformance/fixtures/packets/fp_p2d_024_*`: normalization + unsupported taxonomy fixtures
- `crates/fp-conformance/oracle/pandas_oracle.py`: differential baseline path

## Type Inventory

- `fp_conformance::PacketFixture` fields: `constructor_dtype`, `constructor_copy`
- `fp_types::DType` alias targets: `Bool`, `Int64`, `Float64`, `Utf8`

## Rule Ledger

1. Dtype-spec normalization is case-insensitive and trims surrounding spaces.
2. Alias set is explicitly bounded; unknown specs are rejected.
3. Unsupported dtype specs must emit deterministic error substrings.
4. Alias mapping cannot alter index/column semantics outside dtype coercion.

## Hidden Assumptions

1. Fixture scalar model omits pandas extension-array physical encodings.
2. Unsupported extension specs are represented as fail-closed diagnostic checks.

## Undefined-Behavior Edges

1. Full pandas extension dtype matrix with backend-specific semantics.
2. Complex object/categorical dtype conversions beyond explicit rejection paths.
