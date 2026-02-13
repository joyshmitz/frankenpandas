# Round 1 Baseline

Command benchmarked:

`CARGO_TARGET_DIR=/data/projects/frankenpandas/.target cargo run -p fp-conformance --bin fp-conformance-cli -- --oracle fixture --write-artifacts --require-green`

Hyperfine:

- runs: `20`
- mean: `158.360 ms`
- p50: `150.079 ms`
- p95: `181.833 ms`
- p99: `274.769 ms`
- min: `145.077 ms`
- max: `274.769 ms`

Raw benchmark artifact:

- `artifacts/perf/round1_packet_hyperfine.json`

Syscall profile (`strace -c`):

- top syscall time share: `execve` (`47.31%`)
- read-heavy startup path: `read` (`11.07%`)
- metadata and file lookup pressure: `statx` (`6.62%`)
- process spawning overhead: `clone3` (`7.41%`)

Raw syscall artifact:

- `artifacts/perf/round1_packet_strace.txt`

Interpretation:

- The packet pipeline remains startup and process-launch dominated, now across five packets (`FP-P2C-001`..`FP-P2C-005`) with enforced green gates.
- The next optimization round should target Python oracle spawn amortization and fixture load memoization before kernel-level tuning.
