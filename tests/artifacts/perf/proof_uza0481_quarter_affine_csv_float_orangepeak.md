# br-frankenpandas-uza04.81 Proof - Quarter-Affine CSV Float64 Plan

- Agent: `OrangePeak`
- Head: `217618ecfd25`
- Source lever: `crates/fp-io/src/lib.rs`
- Scope: all-valid typed CSV Float64 writer path only.

## Change

Added a private `Float64QuarterAffineCsvPlan` in `fp-io`.

The plan certifies an all-valid Float64 column as an exact finite nonnegative affine progression in quarter units. Accepted columns emit pandas-compatible fixed decimal text from integer arithmetic. Any uncertainty falls back to the existing per-cell `write_pandas_float` path.

## Behavior Proof

Stable CSV witness is unchanged:

```text
d93c23c308b1cbba82051d7dd022001e84e5313c0b485a440891a604ebfce367  tests/artifacts/perf/uza0481_after_csv_write_read_roundtrip_stable.json
tests/artifacts/perf/uza0481_after_csv_write_read_roundtrip_stable.json: OK
```

Stable fields are byte-identical to the baseline witness:

```json
{
  "checksum": 62503875000.0,
  "io_payload_bytes": 8611370,
  "name": "csv_write_read_roundtrip",
  "rows_out": 100000
}
```

Isomorphism obligations:

- Ordering: row-major loop, column order, header order, and delimiter order unchanged.
- CSV surface: header/index options, quoting, delimiter handling, UTF8 and Int64 paths unchanged.
- Floating point: accepted values are certified by exact `value * 4.0` integer equality; emitted text is derived from that integer numerator without recomputing a float. Unaccepted values keep `write_pandas_float`.
- Fallbacks: negative zero, negative values, NaN, infinities, non-quarter decimals, non-affine series, arithmetic overflow, and too-large scaled values are rejected by the plan.
- Tie-breaking: N/A.
- RNG: N/A.

Focused tests:

```text
CARGO_TARGET_DIR=/data/projects/.scratch/cargo-target-orangepeak-uza0481-check rch exec -- cargo test -p fp-io uza0481 --lib -- --nocapture
CARGO_TARGET_DIR=/data/projects/.scratch/cargo-target-orangepeak-uza0481-check rch exec -- cargo test -p fp-io to_csv_float_format_matches_pandas_str --lib -- --nocapture
```

Both passed. RCH failed open locally for these fast tests because no admissible worker was available at that instant.

Compile/lint/format gates:

```text
CARGO_TARGET_DIR=/data/projects/.scratch/cargo-target-orangepeak-uza0481-check rch exec -- cargo check -p fp-io --all-targets
CARGO_TARGET_DIR=/data/projects/.scratch/cargo-target-orangepeak-uza0481-clippy rch exec -- cargo clippy -p fp-io --all-targets -- -D warnings
cargo fmt -p fp-io -- --check
git diff --check -- crates/fp-io/src/lib.rs .skill-loop-progress.md tests/artifacts/perf/uza0481_pass1_baseline_profile_witness.md tests/artifacts/perf/uza0481_pass2_primitive_selection.md
```

All passed. `cargo check` and `cargo clippy` ran remotely on `vmi1227854`.

UBS:

```text
ubs crates/fp-io/src/lib.rs .skill-loop-progress.md tests/artifacts/perf/uza0481_pass1_baseline_profile_witness.md tests/artifacts/perf/uza0481_pass2_primitive_selection.md tests/artifacts/perf/proof_uza0481_quarter_affine_csv_float_orangepeak.md
```

UBS completed and exited nonzero on pre-existing whole-file `fp-io` inventories: existing panic/unwrap/test/assert/path heuristics, false-positive byte comparisons, and broad allocation/perf inventories. The only new-hunk signal was an informational exact-float certification comparison in `scaled_nonnegative_quarter`; no new UBS critical finding was tied to the quarter-affine plan.

## Benchmark

Workload:

```text
high_ram_perf_baseline --rows 100000 --iters 20 --warmup 3 --frame-cols 10 --key-cardinality 100000
```

`csv_write_read_roundtrip`:

| Metric | Baseline | After | Delta |
| --- | ---: | ---: | ---: |
| mean_ms | 86.04315275 | 51.9041235 | 1.66x faster |
| p50_ms | 86.642249 | 50.023808 | 42.27% lower |
| p95_ms | 92.795965 | 56.942903 | 38.64% lower |
| p99_ms | 94.571176 | 59.312169 | 37.28% lower |
| rows/sec | 1,162,207.5296 | 1,926,629.2012 | 1.66x higher |
| bytes/sec | 100,081,990.5451 | 165,909,169.0470 | 1.66x higher |

Paired command-level hyperfine:

```text
baseline-first: 3.445 s +/- 0.082 -> 2.734 s +/- 0.071, after 1.26x +/- 0.04 faster
after-first:    3.307 s +/- 0.096 -> 2.615 s +/- 0.097, after 1.27x +/- 0.06 faster
```

## Profile

Baseline profile:

- `ryu::pretty::format64`: `32.10%` children / `23.30%` self.
- `fp_io::write_pandas_float`: `8.32%` children / `5.36%` self.

After profile:

- `ryu::pretty::format64`: absent from the sampled CSV hot path.
- `fp_io::write_pandas_float`: absent from the sampled CSV hot path.
- `fp_io::write_csv_string_with_options`: `22.58%` children / `7.35%` self.
- Shifted CSV residual: `core::fmt::write` / integer display under quarter-scaled decimal emission.
- Separate non-CSV residual: `<fp_frame::DataFrameGroupBy>::aggregate_named_func` at `37.17%` children / `1.61%` self.
- `perf record`: `2974` samples, `0` lost samples.

## Score

Impact 4 x Confidence 4 / Effort 3 = `5.33`.

Verdict: KEEP. The lever clears the Score >= 2.0 gate with unchanged golden output and paired/reversed benchmark confirmation.

## Artifacts

- Baseline: `tests/artifacts/perf/uza0481_base_high_ram_keycard100000_100000x20.json`
- After: `tests/artifacts/perf/uza0481_after_highram_keycard100000_100000x20.json`
- Stable SHA: `tests/artifacts/perf/uza0481_after_csv_write_read_roundtrip_stable.sha256`
- Paired hyperfine: `tests/artifacts/perf/uza0481_pair_hyperfine_highram_keycard100000_100000x20.txt`
- Reversed hyperfine: `tests/artifacts/perf/uza0481_pair_reversed_hyperfine_highram_keycard100000_100000x20.txt`
- After profile: `tests/artifacts/perf/uza0481_after_perf_report_children_highram_keycard100000_100000x20.txt`
