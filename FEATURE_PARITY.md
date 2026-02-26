# FEATURE_PARITY

## Status Legend

- not_started
- in_progress
- parity_green
- parity_gap

## Parity Matrix

| Feature Family | Status | Notes |
|---|---|---|
| DataFrame/Series constructors | in_progress | `Series::from_values` + `DataFrame::from_series` MVP implemented; DataFrame constructor scalar-broadcast parity now available via `from_dict_mixed`/`from_dict_with_index_mixed` (fail-closed all-scalar-without-index and mismatched-shape guards); `FP-P2C-003` extends arithmetic fixture coverage; DataFrame `iterrows`/`itertuples`/`items`/`assign`/`pipe` + `where_cond`/`mask` implemented; broader constructor parity pending |
| Expression planner arithmetic | in_progress | `fp-expr` now supports `Expr::Add`/`Sub`/`Mul`/`Div`, logical mask composition (`Expr::And`/`Or`/`Not`), plus `Expr::Compare` (`eq`/`ne`/`lt`/`le`/`gt`/`ge`) with full-eval and incremental-delta paths (including series-scalar anchoring); includes DataFrame-backed eval/query bridges (`EvalContext::from_dataframe`, `evaluate_on_dataframe`, `filter_dataframe_on_expr`); string expression parser with `parse_expr`/`eval_str`/`query_str` for pandas-style `df.eval("a + b")` and `df.query("x > 5 and y < 10")` semantics; broader window/string/date expression surface still pending |
| Index alignment and selection | in_progress | `FP-P2C-001`/`FP-P2C-002` packet suites green with gate validation and RaptorQ sidecars; `FP-P2C-010` adds series/dataframe filter-head-loc-iloc basics; `FP-P2D-025` extends DataFrame loc/iloc row+column selector parity; `FP-P2D-026` adds DataFrame head/tail parity; `FP-P2D-027` adds DataFrame head/tail negative-`n` parity; `FP-P2D-040` adds DataFrame `sort_index`/`sort_values` ordering parity (including descending and NA-last cases); `fp-frame` now also exposes Series `head`/`tail` including negative-`n` semantics, DataFrame `set_index`/`reset_index` (single-index model, including mixed Int64/Utf8 index-label reset materialization), row-level `duplicated`/`drop_duplicates`, and `DataFrame::drop(labels, axis)` for flexible row/column removal plus `drop_rows_int`; broader selector+ordering matrix still pending |
| Series conditional/membership | in_progress | `where_cond`/`mask`/`isin`/`between` implemented with tests; broader conditional matrix pending |
| Series statistics (extended) | in_progress | `idxmin`/`idxmax`/`nlargest`/`nsmallest`/`pct_change`/`corr`/`cov_with`/`prod`/`mode` implemented; `Series::dtype()` accessor added; `dt` accessor with `year`/`month`/`day`/`hour`/`minute`/`second`/`dayofweek`/`date`; `map_fn` failable closure mapping; binary ops with fill_value: `add_fill`/`sub_fill`/`mul_fill`/`div_fill`; `modulo`/`pow` element-wise ops; `nlargest_keep`/`nsmallest_keep` with keep param ('first'/'last'/'all'); broader stats pending |
| Series str accessor | in_progress | `StringAccessor` with `lower`/`upper`/`strip`/`lstrip`/`rstrip`/`contains`/`replace`/`startswith`/`endswith`/`len`/`slice`/`split_get`/`capitalize`/`title`/`repeat`/`pad` plus regex methods: `contains_regex`/`replace_regex`/`replace_regex_all`/`extract`/`count_matches`/`findall`/`fullmatch`/`match_regex`/`split_regex_get`; formatting: `zfill`/`center`/`ljust`/`rjust`; predicates: `isdigit`/`isalpha`/`isalnum`/`isspace`/`islower`/`isupper`/`isnumeric`/`isdecimal`/`istitle`; additional: `get`/`wrap`/`normalize`/`cat`; broader str methods pending |
| DataFrame groupby integration | in_progress | `DataFrame::groupby(&[columns])` returns `DataFrameGroupBy` with `sum`/`mean`/`count`/`min`/`max`/`std`/`var`/`median`/`first`/`last`/`size`/`nunique`/`prod` aggregation; multi-column group keys with composite key support; `agg()` per-column mapping and `agg_list()` multi-function; `apply()` custom closure; `transform()` shape-preserving broadcast; `filter()` group predicate; broader custom function patterns pending |
| DataFrame properties/introspection | in_progress | `shape()`, `dtypes()`, `copy()`, `to_dict(orient)` with dict/list/records/index orients; `info()` string summary; `sample(n, frac, replace, seed)` with deterministic LCG + Fisher-Yates; `compare(other)` element-wise diff |
| Series conversion | in_progress | `to_frame`/`to_list`/`to_dict`/`explode(sep)` implemented; `value_counts_with_options(normalize, sort, ascending, dropna)` full-param variant; `is_unique`/`is_monotonic_increasing`/`is_monotonic_decreasing` introspection |
| Series/DataFrame rank | in_progress | `rank()` with `average`/`min`/`max`/`first`/`dense` methods, `ascending`/`descending`, `na_option` keep/top/bottom; full edge-case matrix pending |
| Rolling/Expanding windows | in_progress | `Series::rolling(window).sum/mean/min/max/std/count()` and `Series::expanding().sum/mean/min/max/std()` plus `DataFrame::rolling(window).sum/mean/min/max/std/count()` and `DataFrame::expanding().sum/mean/min/max/std()` implemented; broader window ops pending |
| DataFrame reshaping | in_progress | `melt(id_vars, value_vars, var_name, value_name)` and `pivot_table(values, index, columns, aggfunc)` implemented with sum/mean/count/min/max/first; `stack`/`unstack` implemented with composite key round-trip; broader reshaping edge cases pending |
| DataFrame aggregation | in_progress | `agg()` with per-column named functions, `applymap()` for element-wise ops, `transform()` shape-preserving variant; column-wise `sum`/`mean`/`min_agg`/`max_agg`/`std_agg`/`var_agg`/`median_agg`/`prod_agg`/`count`/`nunique`/`idxmin`/`idxmax`/`all`/`any` implemented; `apply_row`/`apply_row_fn` for row-wise closures returning Series; broader aggregation patterns pending |
| DataFrame correlation/covariance | in_progress | `corr()`/`cov()` pairwise matrices with Pearson method; `corr_method("spearman")`/`corr_method("kendall")` now implemented including `corr_spearman()`/`corr_kendall()` on Series; broader rank correlation edge cases pending |
| DataFrame element-wise ops | in_progress | `cumsum`/`cumprod`/`cummax`/`cummin`/`diff`/`shift`/`abs`/`clip`/`clip_lower`/`clip_upper`/`round`/`pct_change`/`replace` implemented for DataFrame; `to_csv(sep, include_index)`/`to_json(orient)` string export; delegates to per-column Series methods, non-numeric columns preserved |
| DataFrame selection (extended) | in_progress | `nlargest(n, column)`/`nsmallest(n, column)` for top-N rows, `reindex()` for label-based reindexing, `value_counts_per_column()` implemented; `insert(loc, name, column)` positional column insertion; `pop(name)` remove-and-return column; `align_on_index(other, mode)` DataFrame index alignment; `select_dtypes(include, exclude)` dtype-based column selection; `filter_labels(items, like, regex, axis)` flexible row/column filtering |
| GroupBy core aggregates | in_progress | `FP-P2C-005` and `FP-P2C-011` suites green (`sum`/`mean`/`count` core semantics); `nunique`/`prod`/`size` added for DataFrameGroupBy; `agg()` per-column and `agg_list()` multi-func; `apply()`/`transform()`/`filter()` implemented; broader aggregate matrix still pending |
| Join/merge/concat core | in_progress | `FP-P2C-004` and `FP-P2C-006` suites green for series-level join/concat semantics; `FP-P2D-014` covers DataFrame merge + axis=0 concat matrix; `FP-P2D-028` adds DataFrame concat axis=1 outer alignment parity; `FP-P2D-029` adds axis=1 `join=inner` parity; `FP-P2D-030` adds axis=0 `join=inner` shared-column parity; `FP-P2D-031` adds axis=0 `join=outer` union-column/null-fill parity; `FP-P2D-032` adds axis=0 `join=outer` first-seen column-order (`sort=False`) parity; `FP-P2D-039` adds DataFrame merge `how='cross'` semantics; `DataFrameMergeExt` trait in fp-join adds `merge()`/`merge_with_options()`/`join_on_index()` instance methods; full DataFrame merge/concat contracts still pending |
| Null/NaN semantics | in_progress | `FP-P2C-007` suite green for `dropna`/`fillna`/`nansum`; `fp-frame` now also exposes Series/DataFrame `isna`/`notna` plus `isnull`/`notnull` aliases, DataFrame `fillna`, optioned row-wise `dropna` (`how='any'/'all'` + `thresh` with column `subset` selectors), and optioned column-wise `dropna` (`axis=1`, `how='any'/'all'` + `thresh`, row-label `subset`, plus default `dropna_columns()`); full nanops matrix still pending |
| Core CSV ingest/export | in_progress | `FP-P2C-008` suite green for CSV round-trip core cases; `fp-io` now supports optioned file-based CSV reads (`read_csv_with_options_path`) and JSON `records`/`columns`/`split`/`index` orients (including split index-label roundtrip); broader parser/formatter parity matrix pending |
| Parquet I/O | in_progress | `fp-io` supports `read_parquet`/`write_parquet` (file-based) and `read_parquet_bytes`/`write_parquet_bytes` (in-memory) via Arrow RecordBatch integration; handles Int64/Float64/Bool/Utf8 dtypes with null round-trip; multi-batch reading via `concat_dataframes`; broader Parquet options (compression, row-group control, predicate pushdown) pending |
| Storage/dtype invariants | in_progress | `FP-P2C-009` suite green for dtype invariant checks; `fp-frame` now exposes `Series::astype` plus DataFrame single- and multi-column coercion via `astype_column` and mapping-based `astype_columns`; broader dtype coercion/storage matrix pending |

## Phase-2C Packet Evidence (Current)

| Packet | Result | Evidence |
|---|---|---|
| FP-P2C-001 | parity_green | `artifacts/phase2c/FP-P2C-001/parity_report.json`, `artifacts/phase2c/FP-P2C-001/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-001/parity_report.raptorq.json` |
| FP-P2C-002 | parity_green | `artifacts/phase2c/FP-P2C-002/parity_report.json`, `artifacts/phase2c/FP-P2C-002/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-002/parity_report.raptorq.json` |
| FP-P2C-003 | parity_green | `artifacts/phase2c/FP-P2C-003/parity_report.json`, `artifacts/phase2c/FP-P2C-003/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-003/parity_report.raptorq.json` |
| FP-P2C-004 | parity_green | `artifacts/phase2c/FP-P2C-004/parity_report.json`, `artifacts/phase2c/FP-P2C-004/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-004/parity_report.raptorq.json` |
| FP-P2C-005 | parity_green | `artifacts/phase2c/FP-P2C-005/parity_report.json`, `artifacts/phase2c/FP-P2C-005/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-005/parity_report.raptorq.json` |
| FP-P2C-006 | parity_green | `artifacts/phase2c/FP-P2C-006/parity_report.json`, `artifacts/phase2c/FP-P2C-006/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-006/parity_report.raptorq.json` |
| FP-P2C-007 | parity_green | `artifacts/phase2c/FP-P2C-007/parity_report.json`, `artifacts/phase2c/FP-P2C-007/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-007/parity_report.raptorq.json` |
| FP-P2C-008 | parity_green | `artifacts/phase2c/FP-P2C-008/parity_report.json`, `artifacts/phase2c/FP-P2C-008/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-008/parity_report.raptorq.json` |
| FP-P2C-009 | parity_green | `artifacts/phase2c/FP-P2C-009/parity_report.json`, `artifacts/phase2c/FP-P2C-009/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-009/parity_report.raptorq.json` |
| FP-P2C-010 | parity_green | `artifacts/phase2c/FP-P2C-010/parity_report.json`, `artifacts/phase2c/FP-P2C-010/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-010/parity_report.raptorq.json` |
| FP-P2C-011 | parity_green | `artifacts/phase2c/FP-P2C-011/parity_report.json`, `artifacts/phase2c/FP-P2C-011/parity_gate_result.json`, `artifacts/phase2c/FP-P2C-011/parity_report.raptorq.json` |
| FP-P2D-025 | parity_green | `artifacts/phase2c/FP-P2D-025/parity_report.json`, `artifacts/phase2c/FP-P2D-025/parity_gate_result.json`, `artifacts/phase2c/FP-P2D-025/parity_report.raptorq.json` |
| FP-P2D-026 | parity_green | `artifacts/phase2c/FP-P2D-026/parity_report.json`, `artifacts/phase2c/FP-P2D-026/parity_gate_result.json`, `artifacts/phase2c/FP-P2D-026/parity_report.raptorq.json` |
| FP-P2D-027 | parity_green | `artifacts/phase2c/FP-P2D-027/parity_report.json`, `artifacts/phase2c/FP-P2D-027/parity_gate_result.json`, `artifacts/phase2c/FP-P2D-027/parity_report.raptorq.json` |
| FP-P2D-028 | parity_green | `artifacts/phase2c/FP-P2D-028/parity_report.json`, `artifacts/phase2c/FP-P2D-028/parity_gate_result.json`, `artifacts/phase2c/FP-P2D-028/parity_report.raptorq.json` |
| FP-P2D-029 | parity_green | `artifacts/phase2c/FP-P2D-029/parity_report.json`, `artifacts/phase2c/FP-P2D-029/parity_gate_result.json`, `artifacts/phase2c/FP-P2D-029/parity_report.raptorq.json` |
| FP-P2D-030 | parity_green | `artifacts/phase2c/FP-P2D-030/parity_report.json`, `artifacts/phase2c/FP-P2D-030/parity_gate_result.json`, `artifacts/phase2c/FP-P2D-030/parity_report.raptorq.json` |
| FP-P2D-031 | parity_green | `artifacts/phase2c/FP-P2D-031/parity_report.json`, `artifacts/phase2c/FP-P2D-031/parity_gate_result.json`, `artifacts/phase2c/FP-P2D-031/parity_report.raptorq.json` |
| FP-P2D-032 | parity_green | `artifacts/phase2c/FP-P2D-032/parity_report.json`, `artifacts/phase2c/FP-P2D-032/parity_gate_result.json`, `artifacts/phase2c/FP-P2D-032/parity_report.raptorq.json` |
| FP-P2D-039 | parity_green | `artifacts/phase2c/FP-P2D-039/parity_report.json`, `artifacts/phase2c/FP-P2D-039/parity_gate_result.json`, `artifacts/phase2c/FP-P2D-039/parity_report.raptorq.json` |
| FP-P2D-040 | parity_green | `artifacts/phase2c/FP-P2D-040/parity_report.json`, `artifacts/phase2c/FP-P2D-040/parity_gate_result.json`, `artifacts/phase2c/FP-P2D-040/parity_report.raptorq.json` |

Gate enforcement and trend history:

- blocking command: `./scripts/phase2c_gate_check.sh`
- CI workflow: `.github/workflows/ci.yml`
- drift history ledger: `artifacts/phase2c/drift_history.jsonl`

## Required Evidence Per Feature Family

1. Differential fixture report.
2. Edge-case/adversarial test results.
3. Benchmark delta (when performance-sensitive).
4. Documented compatibility exceptions (if any).
