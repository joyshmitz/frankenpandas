# FrankenPandas

<div align="center">
  <img src="frankenpandas_illustration.webp" alt="FrankenPandas - Clean-room Rust reimplementation of pandas" width="600">

  **Clean-room Rust reimplementation of the full pandas API surface.**

  Drop-in replacement. Zero `unsafe`. Profile-proven performance.

  ![Rust](https://img.shields.io/badge/Rust-2024_edition-orange)
  ![License](https://img.shields.io/badge/license-MIT-blue)
  ![Tests](https://img.shields.io/badge/tests-1500%2B-brightgreen)
  ![IO Formats](https://img.shields.io/badge/IO_formats-7-purple)
</div>

---

## TL;DR

**The Problem:** pandas is the lingua franca of data analysis, but it's single-threaded Python with unpredictable memory spikes, GIL contention in production pipelines, and dtype coercion surprises that silently corrupt results.

**The Solution:** FrankenPandas rebuilds the entire pandas API from first principles in Rust — same semantics, same method names, same edge-case behavior — but with columnar storage, vectorized kernels, arena-backed execution, and compile-time safety guarantees.

**Why FrankenPandas?**

| Feature | pandas | Polars | FrankenPandas |
|---------|--------|--------|---------------|
| API compatibility with pandas | - | Partial (different API) | Full parity target |
| Memory safety | Runtime errors | Safe Rust | Safe Rust (`#![forbid(unsafe_code)]`) |
| Index alignment semantics | Yes | No (no index) | Yes (AACE) |
| GroupBy with named aggregation | Yes | Yes (different syntax) | Yes (`agg_named`) |
| `eval()`/`query()` string expressions | Yes | No | Yes |
| MultiIndex | Yes | No | Yes (foundation) |
| Categorical dtype | Yes | Yes | Yes (metadata layer) |
| 7 IO formats (CSV/JSON/JSONL/Parquet/Excel/SQL/Feather) | Yes | Partial | Yes |
| Conformance testing against pandas oracle | - | - | Yes (20+ packet suites) |

## Quick Example

```rust
use frankenpandas::prelude::*;

// Read CSV (with pandas-style options)
let df = read_csv_str("name,age,city\nAlice,30,NYC\nBob,25,LA\nCarol,35,NYC")?;

// Filter with query expressions (just like df.query() in pandas)
let adults = df.query("age > 28")?;

// GroupBy with named aggregation
let summary = adults.groupby(&["city"])?.agg_named(&[
    ("avg_age", "age", "mean"),
    ("count", "age", "count"),
])?;

// Export to any format
let json = write_json_string(&summary, JsonOrient::Records)?;
let feather = write_feather_bytes(&summary)?;

println!("{}", summary);
// city    avg_age  count
// NYC       32.5      2
```

## Crown-Jewel Innovation

**Alignment-Aware Columnar Execution (AACE):** Every binary operation between DataFrames or Series goes through an explicit index-alignment planning phase before any data is touched. An `EvidenceLedger` records each materialization decision with Bayesian confidence scores, creating a fully auditable execution trace.

This is not a best-effort optimization — it's a core identity constraint. pandas' alignment semantics (outer join on index for arithmetic, left join for assignment) are preserved exactly, with formal correctness evidence.

## Design Philosophy

| Principle | What It Means |
|-----------|---------------|
| **Semantic parity over speed** | We never sacrifice pandas-observable behavior for performance. Every dtype promotion, NaN propagation, and output ordering contract is preserved. |
| **Prove, then optimize** | Each optimization round produces a baseline, an opportunity matrix, an isomorphism proof, and a recommendation contract. No optimization without correctness evidence. |
| **Fail closed** | Unknown features, incompatible dtypes, and ambiguous coercions produce errors, not silent corruption. Strict mode rejects; hardened mode logs and recovers. |
| **Zero unsafe** | Every crate uses `#![forbid(unsafe_code)]`. Memory safety comes from the type system, not audits. |
| **Test everything differentially** | Conformance packets run FrankenPandas operations and compare against the pandas oracle. Machine-readable parity reports catch regressions automatically. |

## Architecture

```
                    ┌─────────────────────┐
                    │   frankenpandas      │  ← Unified facade crate
                    │   (prelude re-exports)│
                    └────────┬────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                     ▼
   ┌─────────┐      ┌──────────────┐      ┌────────────┐
   │ fp-expr  │      │   fp-frame   │      │   fp-io    │
   │ eval()   │      │ DataFrame    │      │ 7 formats  │
   │ query()  │      │ Series       │      │ CSV/JSON/  │
   └────┬─────┘      │ Categorical  │      │ JSONL/     │
        │            │ MultiIndex   │      │ Parquet/   │
        ▼            └──────┬───────┘      │ Excel/SQL/ │
   ┌──────────┐             │              │ Feather    │
   │fp-runtime│      ┌──────┼──────┐       └────────────┘
   │ Policy   │      ▼      ▼      ▼
   │ Ledger   │  fp-index fp-groupby fp-join
   └──────────┘  alignment  arena-   merge_asof
                 planning   backed   cross join
                            agg
        ┌────────────────────┘
        ▼
   ┌───────────┐      ┌──────────────┐
   │fp-columnar│      │  fp-types     │
   │ Column    │      │ Scalar, DType │
   │ ValidMask │      │ NaN/NaT/Null  │
   └───────────┘      └──────────────┘
```

**12 workspace crates**, 84,000+ lines of Rust, 788 public functions.

## Workspace Structure

```
frankenpandas/
├── crates/
│   ├── frankenpandas/    # Unified facade crate with prelude
│   ├── fp-types/         # Scalar, DType, NullKind, coercion rules
│   ├── fp-columnar/      # Column, ValidityMask, vectorized kernels
│   ├── fp-index/         # Index, MultiIndex, alignment planning
│   ├── fp-frame/         # DataFrame, Series, Categorical, 788 methods
│   ├── fp-expr/          # Expression parser, eval()/query()
│   ├── fp-groupby/       # GroupBy with 3 execution paths
│   ├── fp-join/          # Inner/Left/Right/Outer/Cross/Asof joins
│   ├── fp-io/            # 7 IO formats, 95 tests
│   ├── fp-conformance/   # Differential testing against pandas oracle
│   ├── fp-runtime/       # Strict/Hardened policy, EvidenceLedger
│   └── fp-frankentui/    # Terminal UI dashboard (experimental)
├── artifacts/perf/       # Optimization round evidence
└── artifacts/phase2c/    # Conformance packet artifacts
```

## IO Format Support

| Format | Read | Write | In-Memory | File | Options |
|--------|------|-------|-----------|------|---------|
| **CSV** | `read_csv_str` | `write_csv_string` | Yes | Yes | delimiter, headers, na_values, index_col, usecols, nrows, skiprows, dtype |
| **JSON** | `read_json_str` | `write_json_string` | Yes | Yes | 5 orients: Records, Columns, Index, Split, Values |
| **JSONL** | `read_jsonl_str` | `write_jsonl_string` | Yes | Yes | One object per line, blank-line tolerant, union-key detection |
| **Parquet** | `read_parquet_bytes` | `write_parquet_bytes` | Yes | Yes | Arrow RecordBatch integration, multi-batch reading |
| **Excel** | `read_excel_bytes` | `write_excel_bytes` | Yes | Yes | sheet_name, has_headers, index_col, skip_rows; .xlsx/.xls/.xlsb/.ods |
| **Feather** | `read_feather_bytes` | `write_feather_bytes` | Yes | Yes | Arrow IPC file + stream formats |
| **SQL** | `read_sql` | `write_sql` | N/A | SQLite | query or table, SqlIfExists (Fail/Replace/Append), transaction-wrapped |

All formats accessible through `DataFrameIoExt` trait methods on DataFrame.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
frankenpandas = { path = "crates/frankenpandas" }
```

Or use individual crates for finer-grained control:

```toml
[dependencies]
fp-frame = { path = "crates/fp-frame" }
fp-io = { path = "crates/fp-io" }
fp-types = { path = "crates/fp-types" }
```

### Build from Source

```bash
git clone https://github.com/Dicklesworthstone/frankenpandas.git
cd frankenpandas
cargo build --workspace --release
cargo test --workspace
```

Requires **Rust nightly** (2024 edition). See `rust-toolchain.toml`.

## Quick Start

```rust
use frankenpandas::prelude::*;

// 1. Create a DataFrame
let df = read_csv_str("ticker,price,volume\nAAPL,185.50,1000\nGOOG,140.25,500\nAAPL,186.00,1200")?;

// 2. Filter rows
let expensive = df.query("price > 150")?;

// 3. GroupBy and aggregate
let by_ticker = expensive.groupby(&["ticker"])?.sum()?;

// 4. Convert dates
let dates = Series::from_values("d", vec![0i64.into()], vec![Scalar::Utf8("2024-01-15".into())])?;
let parsed = to_datetime(&dates)?;

// 5. Export to multiple formats
let csv = write_csv_string(&by_ticker)?;
let json = write_json_string(&by_ticker, JsonOrient::Records)?;
let feather = write_feather_bytes(&by_ticker)?;

// 6. SQL round-trip
let conn = rusqlite::Connection::open_in_memory()?;
write_sql(&by_ticker, &conn, "results", SqlIfExists::Fail)?;
let back = read_sql_table(&conn, "results")?;
```

## How It Works: Deep Dive

### Data Model

FrankenPandas uses a columnar storage model identical to pandas' internal representation:

```
DataFrame
├── index: Index (Vec<IndexLabel>)  ← Row labels (Int64 or Utf8)
├── columns: BTreeMap<String, Column>  ← Named columns
└── column_order: Vec<String>  ← Insertion-order preservation

Column
├── dtype: DType  ← {Null, Bool, Int64, Float64, Utf8}
├── values: Vec<Scalar>  ← Typed values
└── validity: ValidityMask  ← Bitpacked null bitmap

Scalar = Null(NullKind) | Bool(bool) | Int64(i64) | Float64(f64) | Utf8(String)
NullKind = Null | NaN | NaT  ← Three-way null semantics matching pandas
```

Every `Scalar` knows its own type, and every `Column` enforces type homogeneity through `DType`. The type promotion hierarchy follows pandas exactly: `Null < Bool < Int64 < Float64` (with Utf8 incompatible with numerics).

### ValidityMask: Bitpacked Null Tracking

Nulls are tracked using a packed bitmap, not per-element Option types:

```
ValidityMask {
    words: Vec<u64>,  // Each u64 holds 64 validity bits
    len: usize,       // Actual number of elements
}
```

- **Bit `1`** = valid, **Bit `0`** = null/missing
- Element at position `i` is in word `i / 64`, bit `i % 64`
- Last word is masked to clear unused high bits: `(1u64 << remainder) - 1`
- Boolean algebra: `and_mask`, `or_mask`, `not_mask` operate word-by-word on u64s — 64 null checks per CPU instruction
- `count_valid()` uses `word.count_ones()` (hardware `POPCNT`) across all words

This is the same approach used by Apache Arrow and gives O(n/64) performance for null-aware operations instead of O(n).

### Index Alignment: The AACE Pipeline

When you write `series_a + series_b` in pandas, it silently performs an outer join on the indexes before adding. FrankenPandas makes this explicit:

```
Step 1: Plan
   align_union(left.index, right.index) → AlignmentPlan {
       union_index: Index,          // Merged label set
       left_positions: Vec<Option<usize>>,   // Where each left value goes
       right_positions: Vec<Option<usize>>,  // Where each right value goes
   }

Step 2: Materialize
   Reindex both columns according to the plan.
   None positions → fill with NaN (missing propagation).

Step 3: Execute
   Vectorized element-wise operation on aligned arrays.

Step 4: Log
   EvidenceLedger records the decision: timestamp, mode, action,
   Bayesian posterior, expected losses.
```

The alignment planner uses **adaptive lookup** (AG-13): it detects whether the index is sorted (via a lazily-computed `OnceCell<SortOrder>`) and switches between O(log n) binary search for sorted indexes and O(n) HashMap lookup for unsorted ones.

For multi-way alignment (e.g., `DataFrame.from_series([s1, s2, s3])`), a leapfrog triejoin variant (AG-05) computes the N-way union in a single O(n log n) pass instead of iterative pairwise merges.

### GroupBy: Three Execution Paths

The GroupBy engine automatically selects the fastest execution path based on key cardinality and memory budget:

```
                         ┌──────────────────────┐
                         │  All keys Int64 AND   │
                         │  key range ≤ 65,536?  │
                         └───────┬───────────────┘
                            Yes  │        No
                    ┌────────────┘        └─────────┐
                    ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐
          │ Dense Int64 Path │             │ HashMap Generic  │
          │ O(1) array index │             │ Path (fallback)  │
          │ Pre-alloc by key │             │ (source_idx,sum) │
          │ range: max-min   │             │ pairs, no clone  │
          └─────────────────┘             └─────────────────┘
```

**Path 1 — Dense Int64:** When all group keys are integers spanning ≤ 65,536 values, pre-allocates a dense array indexed directly by `key - min_key`. O(1) per-element grouping with zero hash overhead. Used for common patterns like grouping by year, month, category ID.

**Path 2 — Arena-backed (Bumpalo):** When estimated intermediate memory fits within the arena budget (default 256 MB), allocates all working memory from a Bumpalo bump allocator. Single `malloc` + pointer bumps; bulk deallocation when the arena drops. Zero fragmentation, cache-friendly.

**Path 3 — Global allocator HashMap:** Fallback for arbitrary key types and unbounded cardinality. Stores `(source_index, accumulating_sum)` pairs — never clones the group key Scalar itself (AG-08 optimization). The original IndexLabel is reconstructed at output time from the source position.

All three paths produce identical output. This is verified by property-based tests that run the same inputs through arena and non-arena paths and assert bitwise equality.

### Vectorized Kernels

Arithmetic on typed arrays avoids per-element enum dispatch:

```rust
// Instead of matching each Scalar variant per element:
fn vectorized_binary_f64(
    left: &[f64], right: &[f64],
    left_validity: &ValidityMask, right_validity: &ValidityMask,
    op: fn(f64, f64) -> f64,
) -> (Vec<f64>, ValidityMask)
```

The compiler auto-vectorizes the inner loop to SIMD instructions. Combined validity is computed via `and_mask` on the bitmap words — again, 64 elements per instruction.

When types don't align for the fast path (e.g., Int64 + Float64), a scalar fallback promotes each element through `cast_scalar()`, respecting the full pandas type hierarchy.

### Expression Engine

`df.eval("a + b * c > threshold")` and `df.query("price > 100 and volume < 500")` are powered by a recursive-descent parser in fp-expr:

```
Grammar (simplified):
  expr       → or_expr
  or_expr    → and_expr ( "or" and_expr )*
  and_expr   → comparison ( "and" comparison )*
  comparison → arithmetic ( (">" | "<" | "==" | ...) arithmetic )?
  arithmetic → term ( ("+" | "-") term )*
  term       → atom ( ("*" | "/") atom )*
  atom       → NUMBER | COLUMN_NAME | @LOCAL_VAR | "(" expr ")"
```

The parser produces an `Expr` AST that the evaluator walks, resolving column references against the DataFrame's `EvalContext`. Local variables (prefixed with `@`) are broadcast to Series of the appropriate length. The entire pipeline — parse, resolve, evaluate, filter — happens in a single call with no temporary DataFrames.

### Bayesian Runtime Policy

The runtime distinguishes between **strict mode** (fail on any ambiguity) and **hardened mode** (log and attempt repair). The decision between Allow, Reject, and Repair uses Bayesian expected-loss minimization:

```
For each compatibility issue:
  1. Start with prior P(compatible) based on issue type
     - Unknown feature: 0.25
     - Join admission: 0.60

  2. Update with evidence terms (log-likelihood ratios)
     - allowlist_miss:        compat=-3.5, incomp=-0.2
     - unknown_protocol_field: compat=-2.0, incomp=-0.1
     - overflow_risk:         compat=-0.3, incomp=-1.2

  3. Compute posterior via Bayes' theorem

  4. Calculate expected loss for each action:
     E[loss(Allow)]  = L(allow|compat) * P(compat) + L(allow|incomp) * P(incomp)
     E[loss(Reject)] = L(reject|compat) * P(compat) + L(reject|incomp) * P(incomp)
     E[loss(Repair)] = L(repair|compat) * P(compat) + L(repair|incomp) * P(incomp)

  5. Choose action = argmin expected loss
```

The asymmetric loss matrix penalizes "allow if incompatible" (100.0) far more than "reject if compatible" (6.0), making the system conservative by default. Every decision is recorded in the `EvidenceLedger` with full trace: timestamp, mode, prior, posterior, Bayes factor, and the evidence terms that drove it.

### Categorical Data

Rather than adding a `DType::Categorical` variant (which would require changing 52+ match arms across 4 crates), categoricals are implemented as a metadata layer on Series:

```
Series {
    column: Column<Int64>,  // Stores integer codes (0, 1, 2, ...)
    categorical: Some(CategoricalMetadata {
        categories: Vec<Scalar>,  // ["low", "medium", "high"]
        ordered: bool,            // Whether categories have total ordering
    })
}
```

The `.cat()` accessor provides pandas-compatible operations: `categories()`, `codes()`, `rename_categories()`, `add_categories()`, `remove_unused_categories()`, `set_categories()`, `as_ordered()`, `as_unordered()`, `to_values()`. Missing values use code `-1`.

### MultiIndex

Hierarchical indexing is represented as parallel label vectors (one per level):

```
MultiIndex {
    levels: Vec<Vec<IndexLabel>>,  // levels[0] = ["a","a","b","b"]
    names: Vec<Option<String>>,    //            levels[1] = [1, 2, 1, 2]
}
```

Constructors mirror pandas: `from_tuples`, `from_arrays`, `from_product` (Cartesian product). Operations: `get_level_values(level)`, `droplevel(level)` → `MultiIndexOrIndex`, `swaplevel(i,j)`, `reorder_levels(order)`, `to_flat_index(sep)`.

DataFrame integration via `set_index_multi(&["col1", "col2"], drop, sep)` creates composite index labels, and `to_multi_index(&["col1", "col2"])` extracts a standalone MultiIndex from columns.

### String Accessor

The `.str()` accessor provides 50+ string operations matching pandas `Series.str`:

| Category | Methods |
|----------|---------|
| Case | `lower`, `upper`, `capitalize`, `title`, `casefold`, `swapcase` |
| Whitespace | `strip`, `lstrip`, `rstrip`, `expandtabs` |
| Search | `contains`, `startswith`, `endswith`, `find`, `rfind`, `index_of`, `rindex_of` |
| Transform | `replace`, `slice`, `repeat`, `pad`, `zfill`, `center`, `ljust`, `rjust` |
| Split/Join | `split_get`, `split_count`, `join`, `partition`, `rpartition` |
| Predicates | `isdigit`, `isalpha`, `isalnum`, `isspace`, `islower`, `isupper`, `isnumeric`, `isdecimal`, `istitle` |
| Regex | `contains_regex`, `replace_regex`, `replace_regex_all`, `extract`, `count_matches`, `findall`, `fullmatch`, `match_regex` |
| Prefix/Suffix | `removeprefix`, `removesuffix` |
| Other | `len`, `get`, `wrap`, `normalize`, `cat`, `get_dummies` |

### Join Types

| Type | Behavior | Output Size |
|------|----------|-------------|
| Inner | Only rows with keys in both sides | ≤ min(left, right) |
| Left | All left rows; right fills missing with NaN | = left |
| Right | All right rows; left fills missing with NaN | = right |
| Outer | All rows from both sides | ≤ left + right |
| Cross | Cartesian product (no key matching) | = left × right |
| Asof (backward) | Last right row where `right_key ≤ left_key` | = left |
| Asof (forward) | First right row where `right_key ≥ left_key` | = left |
| Asof (nearest) | Closest right row by absolute distance | = left |

`merge_asof` is particularly useful for time-series data — joining trades with quotes at the nearest preceding timestamp, for example.

### Module-Level Functions

Functions matching pandas top-level API:

| Function | pandas Equivalent | Description |
|----------|-------------------|-------------|
| `concat_dataframes` | `pd.concat` | Concatenate along axis 0 or 1 with join modes |
| `concat_dataframes_with_keys` | `pd.concat(keys=)` | Hierarchical index labeling |
| `to_datetime` | `pd.to_datetime` | Parse dates from strings/epochs |
| `to_timedelta` | `pd.to_timedelta` | Parse durations from strings/seconds |
| `to_numeric` | `pd.to_numeric` | Coerce to numeric with NaN for failures |
| `cut` | `pd.cut` | Equal-width binning |
| `qcut` | `pd.qcut` | Quantile-based binning |
| `timedelta_total_seconds` | `Series.dt.total_seconds()` | Convert timedelta to seconds |

## Performance

Five optimization rounds with formal evidence:

| Round | Optimization | Speedup |
|-------|-------------|---------|
| Round 2 | `align_union` borrowed-key HashMap | Eliminates index clones |
| Round 3 | GroupBy identity-alignment fast path | Skips reindex when indexes match |
| Round 4 | Dense Int64 aggregation path | O(1) array access, no HashMap |
| Round 5 | `has_duplicates` OnceCell memoization | **87% faster** on groupby benchmark |

### Measured Baselines (10K rows, debug profile)

| Operation | p50 | p95 | Notes |
|-----------|-----|-----|-------|
| Join (inner, 50% overlap) | 12ms | 19ms | HashMap-based equijoin |
| Join (outer, 50% overlap) | 27ms | 37ms | Union index construction |
| Boolean filter (5 cols) | 15ms | 20ms | Reindex all columns by mask |
| `head(100)` | 22us | 28us | O(1) slice, no copy |
| Scalar add (5 cols) | 2.5ms | 2.8ms | Per-column vectorized |
| Scalar comparison (5 cols) | 3.0ms | 4.1ms | Per-column with Bool output |

Performance baselines tracked for join (inner/left/right/outer), filter (boolean mask, head/tail), and DataFrame arithmetic at 10K-100K row scales. Benchmarks run via `cargo test -p fp-conformance --test perf_baselines -- --ignored --nocapture`.

## Testing

**1,500+ tests** across the workspace:

| Category | Count | What It Covers |
|----------|-------|----------------|
| fp-frame unit tests | 1,016 | DataFrame, Series, Categorical, MultiIndex integration |
| fp-io tests | 96 | 7 IO formats, adversarial inputs, round-trip correctness |
| fp-index tests | 119 | Index alignment, MultiIndex, duplicate detection |
| fp-join tests | 58 | Inner/Left/Right/Outer/Cross/Asof joins, merge_asof edge cases |
| fp-expr tests | 45 | Expression parsing, eval/query, @local variables |
| fp-conformance tests | 117 | Differential conformance against pandas oracle |
| Property-based (proptest) | 75 | DType coercion, CSV/JSON/SQL/Excel/Feather round-trip, ValidityMask algebra, DataFrame arithmetic invariants |
| End-to-end pipeline | 11 | Full workflow: CSV read → query → groupby → merge → sort → export |
| Performance baselines | 13 | Join/filter/arithmetic latency measurement |

### Conformance Gate

```bash
./scripts/phase2c_gate_check.sh
```

Regenerates conformance packet artifacts and fails closed if any parity report or gate is not green. 20+ packet suites covering alignment, join, groupby, concat, filter, CSV, dtype, null semantics, and more.

## Conformance Testing in Depth

The conformance system is a differential testing framework that verifies FrankenPandas output against the actual pandas library:

```
                  ┌──────────────┐
                  │ Fixture JSON  │  ← Input DataFrame + operation
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
    ┌──────────────────┐   ┌────────────────────┐
    │ FrankenPandas     │   │ pandas Oracle       │
    │ (Rust execution)  │   │ (Python subprocess) │
    └────────┬─────────┘   └────────┬────────────┘
             │                      │
             ▼                      ▼
    ┌──────────────────────────────────────┐
    │         Parity Comparison            │
    │  dtype match? value match? order?    │
    └──────────────┬───────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  ┌──────────────┐   ┌───────────────┐
  │ parity_report │   │ parity_gate   │
  │ .json         │   │ _result.json  │
  └──────────────┘   └───────────────┘
```

**Packet families** cover: series alignment (FP-P2C-001–003), join semantics (FP-P2C-004), groupby aggregates (FP-P2C-005, 011), concat (FP-P2C-006), null/NaN ops (FP-P2C-007), CSV round-trip (FP-P2C-008), dtype invariants (FP-P2C-009), filter/selection (FP-P2C-010), plus 30+ DataFrame-level packets (FP-P2D-014 through FP-P2D-055) covering merge, concat axis options, head/tail, loc/iloc, sort, constructor variants, and more.

Every parity report gets a **RaptorQ repair-symbol sidecar** for bit-rot detection. The drift history ledger (`artifacts/phase2c/drift_history.jsonl`) tracks parity trends over time.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `cargo build` fails with "edition 2024" error | Requires Rust nightly | Install via `rustup toolchain install nightly` and check `rust-toolchain.toml` |
| `Column::from_values` returns `IncompatibleDtypes` | Mixed Utf8 + numeric values in one column | Ensure homogeneous types, or use `to_numeric()` first |
| `query("col > 5")` returns error | Column name not found in DataFrame | Verify column exists with `df.column_names()` |
| CSV round-trip changes Float64 to Int64 | `1.0` written as `"1"`, parsed back as Int64 | Use `dtype` parameter in `CsvReadOptions` to force types |
| SQL write fails with "table already exists" | Default `SqlIfExists::Fail` policy | Use `SqlIfExists::Replace` or `SqlIfExists::Append` |
| Excel round-trip loses integer precision for large values | Excel stores all numbers as f64 | Values within i64 range with zero fraction are recovered as Int64 |
| JSONL reader drops columns | Columns only in later rows were missed | Fixed: reader now unions all keys across all rows |
| `to_datetime` returns NaT | Unrecognized date format | Use `to_datetime_with_format(series, Some("%d/%m/%Y"))` with explicit format |

## Limitations

| Limitation | Status | Workaround |
|-----------|--------|------------|
| No native Datetime dtype | Datetimes stored as ISO 8601 Utf8 strings | Use `to_datetime()` for normalization |
| MultiIndex not yet integrated as DataFrame row index | Foundation type exists | Use `set_index_multi()` for composite keys |
| Categorical metadata not propagated through arithmetic | By design (matches pandas) | Use `.cat().to_values()` to materialize |
| No HDF5, Clipboard, or HTML IO | System-library dependencies | Use Feather (faster) or Parquet instead |
| Single-threaded execution | No parallel execution yet | Profile-proven fast paths compensate |
| No plotting | Requires graphics library | Export to pandas/matplotlib for visualization |

## FAQ

**Q: How compatible is this with pandas?**
A: We target absolute API parity. The same method names, same parameter names, same edge-case behavior. Differential conformance tests verify against the actual pandas oracle. We're in early development so not every method is implemented yet, but the architecture is designed for full coverage.

**Q: Why not just use Polars?**
A: Polars is excellent but has a fundamentally different API (lazy evaluation, no index alignment, different method names). FrankenPandas is for teams that want pandas semantics with Rust performance — no code rewrite required, just change the import.

**Q: Is `unsafe` code used anywhere?**
A: No. Every crate in the workspace uses `#![forbid(unsafe_code)]`. Memory safety comes from the Rust type system.

**Q: How do I use this from Python?**
A: PyO3 bindings are a planned future step. Currently this is a pure Rust library.

**Q: What's the `EvidenceLedger`?**
A: Every alignment decision, dtype coercion, and policy override is logged with Bayesian confidence scores. This creates an auditable trail of exactly how your data was transformed — something pandas silently does without any record.

**Q: What's the performance like?**
A: Five formal optimization rounds with measured evidence. The groupby path alone saw an 87% speedup through memoization and dense aggregation paths. All optimizations include isomorphism proofs showing behavior is unchanged.

**Q: How is the GroupBy implemented?**
A: Three automatic execution paths. Dense Int64 keys (range ≤ 65,536) use O(1) array indexing. Medium cardinality uses Bumpalo arena allocation (single malloc, zero fragmentation). Everything else falls back to a HashMap with source-index referencing to avoid per-group clones. All three produce identical results — verified by property-based tests.

**Q: What does "clean-room" mean?**
A: We never read, reference, or copy from the pandas source code. We study pandas' *behavior* (input → output contracts, edge cases, dtype rules) via the conformance oracle, then implement from first principles in Rust. This avoids any license contamination and often produces better implementations.

**Q: How do you handle NaN vs None vs NaT?**
A: Three distinct null kinds: `NullKind::Null` (generic missing), `NullKind::NaN` (float not-a-number), `NullKind::NaT` (not-a-time). `Float64(NaN)` also counts as missing. `is_missing()` returns true for all of these. `semantic_eq()` treats `NaN == NaN` as true (unlike IEEE 754), matching pandas behavior.

**Q: What's the memory overhead vs pandas?**
A: Comparable for numeric data. `ValidityMask` uses 1 bit per element (vs pandas' 8-byte nullable dtype). `Column` stores typed arrays (`Vec<f64>`, `Vec<i64>`) instead of object arrays. String data uses `Vec<String>` (heap-allocated per element, same as pandas object dtype). Arena-backed GroupBy/Join operations avoid per-group heap fragmentation.

**Q: Can I use this for production ETL pipelines?**
A: The core DataFrame, IO, and GroupBy/Join operations are solid and well-tested (1,500+ tests including adversarial inputs and property-based fuzzing). This is pre-1.0 software — API stability is not yet guaranteed, but the correctness bar is high.

## Key Documents

| Document | Purpose |
|----------|---------|
| `AGENTS.md` | Guidelines for AI coding agents |
| `COMPREHENSIVE_SPEC_FOR_FRANKENPANDAS_V1.md` | Full V1 specification |
| `FEATURE_PARITY.md` | Detailed parity matrix with status per feature family |
| `artifacts/perf/` | Optimization round baselines, opportunity matrices, proofs |
| `artifacts/phase2c/` | Conformance packet artifacts and drift history |

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

MIT License (with OpenAI/Anthropic Rider). See `LICENSE`.
