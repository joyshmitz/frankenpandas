# Join & Filter Performance Baselines

**Date:** 2026-03-15
**Commit:** post-Round5 (frankenpandas-n3t)
**Platform:** Contabo VPS worker (rch fleet), debug profile
**Dataset:** 10,000 rows, 5 columns, 50% index overlap for joins

## Join Baselines (10K rows, 50% overlap)

| Join Type | Mean | p50 | p95 | p99 |
|-----------|------|-----|-----|-----|
| Inner | 0.012954 s | 0.011810 s | 0.019051 s | 0.019483 s |
| Left | 0.015815 s | 0.015888 s | 0.020128 s | 0.020548 s |
| Right | 0.024562 s | 0.023829 s | 0.030459 s | 0.031778 s |
| Outer | 0.027931 s | 0.026582 s | 0.037354 s | 0.038055 s |

## Filter Baselines (10K rows, 5 columns)

| Operation | Mean | p50 | p95 | p99 |
|-----------|------|-----|-----|-----|
| Boolean mask (50% selectivity) | 0.016211 s | 0.015437 s | 0.020119 s | 0.021118 s |
| head(100) | 0.000023 s | 0.000022 s | 0.000028 s | 0.000042 s |

## DataFrame Arithmetic Baselines (10K rows, 5 columns)

| Operation | Mean | p50 | p95 | p99 |
|-----------|------|-----|-----|-----|
| add_scalar(42.0) | 0.002501 s | 0.002471 s | 0.002805 s | 0.002881 s |
| eq_scalar_df(42.0) | 0.003247 s | 0.003041 s | 0.004057 s | 0.004090 s |

## Observations

1. **Inner join is fastest** (~13ms for 10K), outer join is slowest (~28ms). This matches expectations since inner produces fewer output rows.
2. **Right join (~25ms) is slower than left join (~16ms)**. This suggests the probe-phase implementation has an asymmetry worth investigating in future optimization rounds.
3. **Boolean mask filter (~16ms)** is comparable to join costs at the same row count. This is dominated by the reindex operation.
4. **head() is O(1)** (~23μs) as expected - it doesn't copy data, just slices.
5. **Scalar arithmetic (~2.5ms)** is ~6x faster than joins, confirming element-wise ops don't pay alignment cost.

## Benchmark Command

```bash
cargo test -p fp-conformance --test perf_baselines perf_run_all_baselines -- --ignored --nocapture
```

## Opportunity Matrix

| Bottleneck | Current Cost | Estimated Savings | Lever |
|-----------|-------------|-------------------|-------|
| Right join probe asymmetry | ~25ms vs ~16ms (left) | ~35% | Optimize right-join to mirror left-join probe |
| Boolean mask reindex | ~16ms | ~50% | Vectorized position extraction instead of per-row |
| Join HashMap build | ~10ms (estimated) | ~20% | Pre-sorted merge-join for sorted indices |
| Scalar comparison dispatch | ~3ms | ~30% | Vectorized typed-array comparison path |
