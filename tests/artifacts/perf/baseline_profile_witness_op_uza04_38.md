# br-frankenpandas-uza04.38 Baseline/Profile Witness

Scope: measurement-only pass for the dense inner-join output lifecycle residual after
`br-frankenpandas-muis1`. No source files were edited.

## Build

- RCH command:
  `env CARGO_TARGET_DIR=/data/projects/.scratch/cargo-target-orangepeak-uza04-38-baseline-20260607T0355Z RUSTFLAGS='-C force-frame-pointers=yes' rch exec -- cargo build -p fp-conformance --profile release-perf --example perf_profile`
- Result: success on RCH worker `ts1`.
- Binary:
  `/data/projects/.scratch/cargo-target-orangepeak-uza04-38-baseline-20260607T0355Z/release-perf/examples/perf_profile`
- Build log:
  `tests/artifacts/perf/build_perf_profile_op_uza04_38.txt`
- Source revision:
  `eb741181e9f6c551177dcebdd29680cfac3d64ba`

## Golden Outputs

Golden mode uses deterministic `n=5000` output dumps, matching the existing join
proof scale. Timing/profile workloads use the requested `n=100000` scenarios.

- `tests/artifacts/perf/golden_before_inner_join_op_uza04_38.txt`
  - sha256: `be7d17114e5a2a88607bdc65228751789e208ebbd9ac2f5d0c616cdf64e641f1`
- `tests/artifacts/perf/golden_before_join_1to1_op_uza04_38.txt`
  - sha256: `18988b22befdb71941112638f6f1c3415cbf0f79fc1f00ecd013318d924682c7`
- Checksum file:
  `tests/artifacts/perf/golden_before_join_outputs_op_uza04_38.sha256`
- Verification output:
  `tests/artifacts/perf/golden_before_join_outputs_op_uza04_38.verify.txt`

## Timing Panel

Command:
`hyperfine --warmup 3 --runs 10 --export-json tests/artifacts/perf/fp_baseline_op_uza04_38.json "<bin> inner_join 100000 3" "<bin> join_1to1 100000 20"`

RCH note: the binary was RCH-built. A probe through `rch exec` for non-compilation
commands emitted a warning and ran on the local host, so the hyperfine execution
itself was local. See `tests/artifacts/perf/rch_probe_hyperfine_op_uza04_38.txt`.

Results:

| Scenario | Mean | Stddev | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_join 100000 3` | 126.060 ms | 9.175 ms | 125.189 ms | 111.349 ms | 144.144 ms |
| `join_1to1 100000 20` | 34.635 ms | 2.074 ms | 34.747 ms | 30.520 ms | 37.810 ms |

Artifacts:

- `tests/artifacts/perf/fp_baseline_op_uza04_38.json`
- `tests/artifacts/perf/hyperfine_baseline_op_uza04_38.txt`

## CPU Profile

Fresh profile command:
`perf record -F 999 --call-graph fp -g -o tests/artifacts/perf/perf_inner_join_op_uza04_38.data -- <bin> inner_join 100000 3`

Artifacts:

- `tests/artifacts/perf/perf_inner_join_op_uza04_38.data`
- `tests/artifacts/perf/perf_record_inner_join_op_uza04_38.txt`
- `tests/artifacts/perf/perf_report_inner_join_op_uza04_38.txt`

Fresh top frames:

- `__memmove_avx_unaligned_erms`: 73.56% children, 5.53% self.
- Unresolved kernel frame `0xffffffff8621b2b7`: 15.30% self.
- Unresolved kernel frame `0xffffffff84c01192`: 10.08% self.
- Kernel symbol maps were restricted, so many kernel frames were unresolved.

Prior named profile context from
`tests/artifacts/perf/perf_report_after_muis1_right_tape.txt`:

- `fp_join::build_dense_i64_inner_output_data::{closure...}`: 9.22% self.
- `__memmove_avx_unaligned_erms`: 6.57% self, with additional child samples.
- `core::ptr::drop_in_place::<fp_columnar::Column>` via `__munmap`: visible in
  two top-level stacks.

## Target Status

The target remains profile-backed for one `crates/fp-join/src/lib.rs` lever. The
fresh profile still points at dense output memory movement, and the prior named
post-muis1 report ties that residual to the dense inner-join output worker plus
`Column` drop/materialization lifecycle. A next pass should use one fp-join-only
output lifecycle lever and preserve left probe order, right bucket insertion
order, duplicate tie-breaking, column order/names/dtypes, null/fallback behavior,
floating-point bits, RNG state, and hash/tie behavior.
