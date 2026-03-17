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

## Performance

Five optimization rounds with formal evidence:

| Round | Optimization | Speedup |
|-------|-------------|---------|
| Round 2 | `align_union` borrowed-key HashMap | Eliminates index clones |
| Round 3 | GroupBy identity-alignment fast path | Skips reindex when indexes match |
| Round 4 | Dense Int64 aggregation path | O(1) array access, no HashMap |
| Round 5 | `has_duplicates` OnceCell memoization | **87% faster** on groupby benchmark |

Performance baselines tracked for join (inner/left/right/outer), filter (boolean mask, head/tail), and DataFrame arithmetic at 10K-100K row scales.

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
