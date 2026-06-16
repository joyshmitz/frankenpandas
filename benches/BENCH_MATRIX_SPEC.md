# FrankenPandas vs Pandas Benchmark Matrix Specification

**Bead:** br-frankenpandas-rg8ys.7.1
**Version:** v3
**Date:** 2026-05-25

## Purpose

Define comprehensive head-to-head benchmark coverage to validate (or disprove) the claim that FrankenPandas "exceeds pandas across the board."

## Category Weights

| Category | Weight | Rationale |
|----------|--------|-----------|
| IO | 0.25 | Dominates real-world pipeline time |
| DataFrameOps | 0.20 | Core transformations, filtering, sorting |
| GroupBy | 0.20 | Analytics cornerstone |
| Joins | 0.15 | Common in ETL pipelines |
| Rolling/Expanding | 0.10 | Time-series analytics |
| Indexing | 0.10 | Label/position access patterns |

## Workload Matrix

### 1. IO (weight: 0.25)

| Workload | Description | Sizes |
|----------|-------------|-------|
| csv_read | Parse CSV to DataFrame | 10k/100k/1M |
| csv_write | DataFrame to CSV | 10k/100k/1M |
| parquet_read | Columnar parquet read | 10k/100k/1M |
| parquet_write | DataFrame to parquet | 10k/100k/1M |
| json_read_records | Parse JSON records | 10k/100k/1M |
| json_write_records | DataFrame to JSON | 10k/100k/1M |

### 2. DataFrameOps (weight: 0.20)

| Workload | Description | Sizes |
|----------|-------------|-------|
| sort_values_single | Single column sort | 10k/100k/1M |
| sort_values_multi | Multi-column sort | 10k/100k/1M |
| filter_bool_mask | Boolean mask selection | 10k/100k/1M |
| drop_duplicates | Deduplicate rows | 10k/100k/1M |
| value_counts | Frequency count | 10k/100k/1M |
| cumsum | Cumulative sum | 10k/100k/1M |

### 3. GroupBy (weight: 0.20)

| Workload | Description | Sizes |
|----------|-------------|-------|
| groupby_sum_int64 | Sum aggregation, dense keys | 10k/100k/1M |
| groupby_mean_float64 | Mean with nulls | 10k/100k/1M |
| groupby_agg_multi | Multiple aggregations | 10k/100k/1M |
| groupby_transform | Transform within groups | 10k/100k/1M |
| groupby_apply | Custom function apply | 10k/100k/1M |
| groupby_ngroup | Group numbering | 10k/100k/1M |

### 4. Joins (weight: 0.15)

| Workload | Description | Sizes |
|----------|-------------|-------|
| merge_inner | Inner join on key | 10k/100k/1M |
| merge_left | Left join | 10k/100k/1M |
| merge_outer | Outer join | 10k/100k/1M |
| merge_asof | Time-series asof join | 10k/100k |
| concat_axis0 | Vertical concatenation | 10k/100k/1M |
| concat_axis1 | Horizontal concatenation | 10k/100k |

### 5. Rolling/Expanding (weight: 0.10)

| Workload | Description | Sizes |
|----------|-------------|-------|
| rolling_mean_w10 | Rolling mean, window=10 | 10k/100k/1M |
| rolling_std_w50 | Rolling std, window=50 | 10k/100k/1M |
| expanding_sum | Expanding sum | 10k/100k/1M |
| ewm_mean | Exponential weighted mean | 10k/100k/1M |

### 6. Indexing (weight: 0.10)

| Workload | Description | Sizes |
|----------|-------------|-------|
| iloc_slice | Position-based slice | 10k/100k/1M |
| loc_labels | Label-based selection | 10k/100k/1M |
| at_scalar | Single cell access | 10k/100k/1M |
| reindex | Index alignment | 10k/100k/1M |

## Size Configurations

| Size | Rows | Columns | Description |
|------|------|---------|-------------|
| 10k | 10,000 | 10 | Small dataset, hot cache |
| 100k | 100,000 | 10 | Medium dataset |
| 1M | 1,000,000 | 10 | Large dataset, memory pressure |

## Dtype Variants

Each numeric workload runs with:
- `int64`: Pure integer column
- `float64`: Floating point with no nulls
- `float64_nan10`: 10% NaN density
- `float64_nan50`: 50% NaN density

## JSON Result Schema (v3)

```json
{
  "schema_version": "v3",
  "engine": {
    "name": "frankenpandas|pandas",
    "version": "0.1.0|2.2.3",
    "platform": "linux-x86_64",
    "compiled_flags": ["release", "lto"]
  },
  "timestamp": "2026-05-25T12:00:00Z",
  "results": [
    {
      "category": "io",
      "workload": "csv_read",
      "size": "100k",
      "dtype": "float64",
      "iterations": 100,
      "metrics": {
        "p50_us": 12500,
        "p95_us": 14200,
        "p99_us": 15800,
        "mean_us": 12800,
        "stddev_us": 890,
        "cv_pct": 6.95,
        "throughput_rows_sec": 7812500
      }
    }
  ],
  "summary": {
    "total_workloads": 42,
    "total_iterations": 4200,
    "weighted_score": 1.0
  }
}
```

## Scoring Formula

```
category_score = geomean(workload_ratios in category)
  where workload_ratio = pandas_p50 / frankenpandas_p50

weighted_score = sum(category_score * category_weight)

claim_validated = weighted_score > 1.0 for ALL categories
```

## Reporting

### Per-Category Scorecard

```
| Category      | Weight | FP p50  | PD p50  | Ratio | Verdict |
|---------------|--------|---------|---------|-------|---------|
| IO            | 0.25   | 12.5ms  | 45.2ms  | 3.6x  | FASTER  |
| DataFrameOps  | 0.20   | 8.2ms   | 15.1ms  | 1.8x  | FASTER  |
| GroupBy       | 0.20   | 5.1ms   | 6.8ms   | 1.3x  | FASTER  |
| Joins         | 0.15   | 22.4ms  | 18.9ms  | 0.8x  | SLOWER  |
| Rolling       | 0.10   | 3.2ms   | 4.1ms   | 1.3x  | FASTER  |
| Indexing      | 0.10   | 0.5ms   | 0.8ms   | 1.6x  | FASTER  |
|---------------|--------|---------|---------|-------|---------|
| WEIGHTED      | 1.00   |    -    |    -    | 1.82x | PARTIAL |
```

### Verdict Categories

- **FASTER**: FP is >1.05x faster (exceeds pandas)
- **PARITY**: FP is 0.95x-1.05x (equivalent)
- **SLOWER**: FP is <0.95x (pandas wins)

## DoD Checklist

- [x] Spec committed with 6 categories, weights, workloads
- [x] JSON v3 schema defined
- [x] Scoring formula documented
- [ ] Reviewed before T7_HARNESS implementation
