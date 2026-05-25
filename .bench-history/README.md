# Benchmark History

This directory stores benchmark baselines for the performance ratchet gate.

## Files

- `latest.json` - The current committed baseline for CI gates
- `<commit-sha>.json` - Historical baselines for specific commits

## Ratchet Thresholds

Per the gauntlet spec:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| primary | -3% | Any single benchmark p50 |
| geomean | -5% | Category-level geometric mean |
| per-category | -10% | Category weighted score |
| p90 | -15% | Tail latency (p95) |
| throughput | -5% | Rows/second |

## Usage

```bash
# Compare new results against baseline
python scripts/perf_ratchet.py --baseline .bench-history/latest.json --new artifacts/bench/current.json

# Update baseline with new results
python scripts/perf_ratchet.py --update-baseline artifacts/bench/current.json
```

## Verdicts

- **ALLOW**: All thresholds pass, OK to update baseline
- **BLOCK**: Regression beyond threshold, CI fails
- **QUARANTINE**: High cv measurements, needs manual review
