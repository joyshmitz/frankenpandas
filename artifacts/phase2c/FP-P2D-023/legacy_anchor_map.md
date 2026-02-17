# FP-P2D-023 Legacy Anchor Map

Packet: `FP-P2D-023`
Subsystem: DataFrame constructor dtype/copy parity

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/frame.py` (`DataFrame.__init__` dtype/copy constructor pathways)
- `legacy_pandas_code/pandas/pandas/core/internals/construction.py` (dtype coercion and constructor casting behavior)
- `legacy_pandas_code/pandas/pandas/core/dtypes/cast.py` (cast failure taxonomy)

## Extracted Behavioral Contract

1. Constructor dtype requests apply deterministic per-column coercion on constructor output.
2. Invalid casts fail closed and surface explicit cast diagnostics.
3. Unsupported dtype specifications fail with deterministic error messages.
4. `copy` flag acceptance does not alter observable constructor output shape/values in pure fixture replay.

## Rust Slice Implemented

- `crates/fp-conformance/src/lib.rs`: constructor option parsing (`constructor_dtype`, `constructor_copy`) and post-construction coercion path
- `crates/fp-conformance/oracle/pandas_oracle.py`: constructor replay path used for differential parity
- `crates/fp-columnar/src/lib.rs`: dtype-aware column coercion primitives

## Type Inventory

- `fp_conformance::PacketFixture` fields: `constructor_dtype`, `constructor_copy`
- `fp_conformance::FixtureOperation`: constructor operation matrix (`dataframe_from_*`, constructor variants)
- `fp_types::DType`: `Bool`, `Int64`, `Float64`, `Utf8`

## Rule Ledger

1. Constructor dtype coercion is optional and operation-agnostic for supported constructor fixtures.
2. Unsupported dtype specs are rejected before constructor cast execution.
3. Cast failures propagate deterministic diagnostics used by error fixtures.
4. Copy flag is accepted and tracked without mutating constructor output contracts.

## Hidden Assumptions

1. Fixture scalar model does not encode pandas extension-array payload internals.
2. Copy semantics are validated through output parity, not mutation-observability probes.

## Undefined-Behavior Edges

1. pandas object dtype fallback semantics across mixed scalar/object payloads.
2. Extension-array constructor behavior requiring richer scalar/array fixture encodings.
