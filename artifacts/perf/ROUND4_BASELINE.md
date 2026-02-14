# ROUND4 Baseline

Command:

```bash
.target-opt/release/groupby-bench --rows 100000 --key-cardinality 512 --iters 30
```

Baseline (`round4_groupby_hyperfine_before.json`):
- mean: `0.292843960 s`
- p50: `0.292189986 s`
- p95: `0.301201611 s`
- p99: `0.306794720 s`

Post-lever (`round4_groupby_hyperfine_after.json`):
- mean: `0.290649768 s`
- p50: `0.289520563 s`
- p95: `0.299307571 s`
- p99: `0.300423827 s`

Delta:
- mean latency improvement: `0.75%`
