# br-frankenpandas-uza04.40 pass 1/5 baseline/profile witness

Mission: baseline and profile witness only. No source edits, no bead close, no
commit.

## Context

- Repo: `/data/projects/frankenpandas`
- Git commit: `6b2d4769f2f4c81de1016e9ba8eaac929b516743`
- Bead: `br-frankenpandas-uza04.40`
- Target: dense inner-join output assembly residual after repeated-slice lanes
- RCH worker: `ts1`
- Local retrieved target dir:
  `/data/projects/.scratch/cargo-target-orangepeak-uza04-40-pass1-20260607T181058Z`
- Binary:
  `/data/projects/.scratch/cargo-target-orangepeak-uza04-40-pass1-20260607T181058Z/release-perf/examples/perf_profile`

## Commands

```bash
CARGO_TARGET_DIR=/data/projects/.scratch/cargo-target-orangepeak-uza04-40-pass1-20260607T181058Z \
RUSTFLAGS='-C force-frame-pointers=yes' \
rch exec -- cargo build -p fp-conformance --profile release-perf --example perf_profile

/data/projects/.scratch/cargo-target-orangepeak-uza04-40-pass1-20260607T181058Z/release-perf/examples/perf_profile golden inner_join 100000 \
  > tests/artifacts/perf/golden_before_inner_join_op_uza04_40.txt

/data/projects/.scratch/cargo-target-orangepeak-uza04-40-pass1-20260607T181058Z/release-perf/examples/perf_profile golden join_1to1 100000 \
  > tests/artifacts/perf/golden_before_join_1to1_op_uza04_40.txt

sha256sum tests/artifacts/perf/golden_before_inner_join_op_uza04_40.txt \
  tests/artifacts/perf/golden_before_join_1to1_op_uza04_40.txt \
  > tests/artifacts/perf/golden_before_join_outputs_op_uza04_40.sha256

sha256sum -c tests/artifacts/perf/golden_before_join_outputs_op_uza04_40.sha256 \
  > tests/artifacts/perf/golden_before_join_outputs_op_uza04_40.verify.txt

hyperfine --warmup 3 --runs 10 \
  --export-json tests/artifacts/perf/fp_baseline_op_uza04_40.json \
  --command-name inner_join_100000x3 \
  '/data/projects/.scratch/cargo-target-orangepeak-uza04-40-pass1-20260607T181058Z/release-perf/examples/perf_profile inner_join 100000 3' \
  --command-name join_1to1_100000x20 \
  '/data/projects/.scratch/cargo-target-orangepeak-uza04-40-pass1-20260607T181058Z/release-perf/examples/perf_profile join_1to1 100000 20' \
  --command-name inner_join_read_100000x3 \
  '/data/projects/.scratch/cargo-target-orangepeak-uza04-40-pass1-20260607T181058Z/release-perf/examples/perf_profile inner_join_read 100000 3'

perf record -F 999 -g --call-graph fp \
  -o tests/artifacts/perf/perf_inner_join_op_uza04_40.data -- \
  /data/projects/.scratch/cargo-target-orangepeak-uza04-40-pass1-20260607T181058Z/release-perf/examples/perf_profile inner_join 100000 3

perf report -i tests/artifacts/perf/perf_inner_join_op_uza04_40.data \
  --stdio --sort comm,dso,symbol --percent-limit 0.5 \
  > tests/artifacts/perf/perf_report_inner_join_op_uza04_40.txt
```

## Build

- Artifact: `tests/artifacts/perf/build_perf_profile_op_uza04_40.txt`
- RCH selected worker: `ts1 at ubuntu@192.168.1.107`
- Remote build result: exit 0
- Remote command time: `135170ms`
- RCH end-to-end line: `[RCH] remote ts1 (149.9s)`

## Golden Outputs

Golden mode supports `inner_join` and `join_1to1`; `inner_join_read` is a
forced-materialization timing scenario but has no separate golden mode in the
current harness.

- `golden_before_inner_join_op_uza04_40.txt`: `976454217` bytes
- `golden_before_join_1to1_op_uza04_40.txt`: `4955812` bytes
- `golden_before_join_outputs_op_uza04_40.verify.txt`: both `OK`

Hashes:

```text
494106fca6e3310a318f1685c74041a2788089a4d2409107d4eef4a00c7a0764  tests/artifacts/perf/golden_before_inner_join_op_uza04_40.txt
102690aa39952cc2d13bcc41547aacdeac1946113e43d62472fdb93440bc56a7  tests/artifacts/perf/golden_before_join_1to1_op_uza04_40.txt
```

## Hyperfine Baseline

Artifact pair:

- `tests/artifacts/perf/fp_baseline_op_uza04_40.json`
- `tests/artifacts/perf/hyperfine_baseline_op_uza04_40.txt`

Results:

| Scenario | Mean | Stddev | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `inner_join 100000 3` | `46.017 ms` | `3.260 ms` | `41.660 ms` | `52.792 ms` |
| `join_1to1 100000 20` | `53.868 ms` | `9.987 ms` | `40.034 ms` | `73.844 ms` |
| `inner_join_read 100000 3` | `384.665 ms` | `17.608 ms` | `362.507 ms` | `418.443 ms` |

## CPU Profile

Artifacts:

- `tests/artifacts/perf/perf_inner_join_op_uza04_40.data`
- `tests/artifacts/perf/perf_record_inner_join_op_uza04_40.txt`
- `tests/artifacts/perf/perf_report_inner_join_op_uza04_40.txt`

Profile command: `inner_join 100000 3`

Perf captured `93` samples. Kernel symbols were partially restricted by host
policy (`kptr_restrict` / `perf_event_paranoid` warning), but the user-space
join residual is visible.

Top visible symbols / frames:

- `fp_join::build_dense_i64_inner_output_data`: `26.19%` children, `7.85%` self
- `__memset_avx2_unaligned_erms`: `23.36%` children
- `core::ptr::drop_in_place::<fp_columnar::Column>`: `8.21%` children, `2.64%` self
- `fp_join::build_single_key_dense_i64_inner_merge_output`: `7.69%` self
- `<alloc::raw_vec::RawVecInner>::finish_grow` / `realloc` / `__memmove_avx_unaligned_erms`: `2.90%`
- `<fp_columnar::Column>::from_i64_repeat_runs`: `2.70%` self
- `<fp_columnar::ColumnData>::from_scalars`: `2.10%` children
- `<fp_columnar::Column>::new`: `2.07%` self

## Residual Candidates

1. Output assembly lifecycle remains the top named user-space residual:
   `build_dense_i64_inner_output_data` plus
   `build_single_key_dense_i64_inner_merge_output`.
2. Allocation/zeroing/drop pressure is still present after the repeated-slice
   lane keep: `memset`, `RawVecInner::finish_grow`, `realloc`, `Column::new`,
   and `drop_in_place::<Column>`.
3. Forced read is the downstream-consumer gate: `inner_join_read 100000 3`
   costs `384.665 ms`, about `8.36x` the lazy `inner_join 100000 3` baseline.

Next pass should profile the output assembly plan deeply before choosing a
single lever; the likely structural primitive is a shared gather/segment-view
materialization plan that reduces per-column construction/drop/zero-fill churn
without changing ordering, duplicate tie-breaking, dtype, or null semantics.
