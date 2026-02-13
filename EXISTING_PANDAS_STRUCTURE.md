# EXISTING_PANDAS_STRUCTURE

## 1. Legacy Oracle

- Root: /dp/frankenpandas/legacy_pandas_code/pandas
- Upstream: pandas-dev/pandas

## 2. Subsystem Map

- pandas/api: public API export surface, extension hooks, interchange and type APIs.
- pandas/core: primary runtime semantics.
  - frame.py and series.py for top-level container behavior.
  - indexes/ for label model and alignment.
  - dtypes/ and arrays/ for inference, casting, extension arrays.
  - internals/blocks.py and internals/managers.py for storage management.
  - indexing.py and indexers/ for loc/iloc behavior.
  - reshape/, groupby/, window/, ops/, missing.py, nanops.py for analytics semantics.
- pandas/_libs: C-accelerated internals for hot algorithm paths.
- pandas/io: parsers and format adapters.
- pandas/tests and pandas/_testing: conformance baseline.

## 3. Semantic Hotspots (Must Preserve)

1. DataFrame constructor alignment and copy/view behavior from frame.py.
2. Series index union alignment during arithmetic and comparisons.
3. Index object behavior for hashability, slicing, get_indexer, and MultiIndex tuple semantics.
4. BlockManager invariants:
   - blocks and axes consistency
   - blknos/blklocs mapping integrity
5. loc/iloc distinction:
   - loc is label-based with missing-label rules
   - iloc is positional only
6. GroupBy/window defaults (dropna/observed/min_periods/closed/center) and aggregation ordering.
7. Arithmetic/operator dispatch alignment and dtype promotion.

## 4. Compatibility-Critical Behaviors

- Assignment and reindex interactions under setitem/insert.
- Chained-assignment warning/error surfaces in indexing path.
- Concat/merge/reindex behavior around missing labels and duplicate keys.
- ExtensionArray interoperability in constructor and arithmetic pipelines.

## 5. Security and Stability Risk Areas

- Expression evaluation surfaces in core/computation (string-based expression execution).
- SQL adapters in io/sql.py (parameterization and injection risk).
- Pickle and other binary formats in io/.
- CSV parsing resource exhaustion and malformed quoting edge cases.
- Heavy third-party IO dependencies requiring explicit trust boundaries.

## 6. V1 Extraction Boundary

Include now:
- core DataFrame/Series/indexing/storage/dtype/ops/reshape/groupby/window semantics.

Exclude for V1:
- full io ecosystem breadth, plotting, docs/web/tooling, direct C-internals replication beyond required behavior.

## 7. High-Value Conformance Fixture Families

- tests/frame: constructor, assign, setitem behavior.
- tests/series and tests/indexes: scalar/index/dtype contracts.
- tests/indexing: loc/iloc/boolean edge semantics.
- tests/groupby and tests/window: aggregation and rolling semantics.
- tests/reshape and tests/arithmetic: merge/concat and binary-op parity.
- tests/arrays and tests/extension: extension storage and API contracts.

## 8. Extraction Notes for Rust Spec

- Treat core/internals + core/indexing + core/indexes as first-order compatibility contract.
- Preserve observable behavior before replacing internal storage strategy.
- Build parity reports per feature family before performance work.
