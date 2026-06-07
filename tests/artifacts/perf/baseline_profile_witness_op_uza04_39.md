# br-frankenpandas-uza04.39 Baseline/Profile Witness

## Source

- Source revision: `54c49d101c531b780444d2947f305b59b2d512c3`
- RCH worker: `ts1` at `ubuntu@192.168.1.107`
- Target dir: `/data/projects/.scratch/cargo-target-orangepeak-uza04-39-baseline-20260607T043122Z`
- Binary: `/data/projects/.scratch/cargo-target-orangepeak-uza04-39-baseline-20260607T043122Z/release-perf/examples/perf_profile`
- Build log: `tests/artifacts/perf/build_perf_profile_op_uza04_39.txt`

## Golden Outputs

`sha256sum -c tests/artifacts/perf/golden_before_join_outputs_op_uza04_39.sha256` passed.

| Scenario | File | SHA-256 |
| --- | --- | --- |
| `golden inner_join 5000` | `tests/artifacts/perf/golden_before_inner_join_op_uza04_39.txt` | `be7d17114e5a2a88607bdc65228751789e208ebbd9ac2f5d0c616cdf64e641f1` |
| `golden join_1to1 5000` | `tests/artifacts/perf/golden_before_join_1to1_op_uza04_39.txt` | `18988b22befdb71941112638f6f1c3415cbf0f79fc1f00ecd013318d924682c7` |

## Hyperfine Baseline

Command: `hyperfine --warmup 3 --runs 10 --export-json tests/artifacts/perf/fp_baseline_op_uza04_39.json ...`

| Scenario | Mean | Stddev | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_join 100000 3` | 136.204 ms | 8.032 ms | 137.568 ms | 123.994 ms | 148.474 ms |
| `join_1to1 100000 20` | 42.704 ms | 5.919 ms | 42.164 ms | 35.288 ms | 50.503 ms |
| `inner_join_read 100000 3` | 416.320 ms | 35.173 ms | 408.255 ms | 389.991 ms | 509.955 ms |

All three rows used `100000` and the requested run count; no reduced `inner_join_read` fallback was needed.

## CPU Profile

Command: `perf record -F 999 --call-graph fp -g -o tests/artifacts/perf/perf_inner_join_op_uza04_39.data -- <bin> inner_join 100000 3`

- Record log: `tests/artifacts/perf/perf_record_inner_join_op_uza04_39.txt`
- Report: `tests/artifacts/perf/perf_report_inner_join_op_uza04_39.txt`
- Samples: 840 cycles samples, 0 lost samples.
- Host caveat: kernel address maps are restricted, so many kernel frames are unresolved.

Top relevant report entries:

| Children | Self | Symbol |
| ---: | ---: | --- |
| 76.36% | 5.24% | `__memmove_avx_unaligned_erms` |
| 13.35% | 0.00% | `core::ptr::drop_in_place::<fp_columnar::Column>` |
| 12.84% | 0.00% | `__munmap` |
| 3.47% | 2.33% | `std::sys::backtrace::__rust_begin_short_backtrace::<fp_join::build_dense_i64_inner_output_data...>` |
| 3.36% | 0.77% | `fp_join::build_dense_i64_inner_output_data` |
| 2.35% | 0.31% | `__memset_avx2_unaligned_erms` |
| 1.80% | 0.00% | `perf_profile::build_join_frame_offset` |
| 1.16% | 0.00% | `<fp_columnar::ColumnData>::from_scalars` |

## Target Rationale

The target remains profile-backed for a lazy bucket-tape `Column` implementation. The dense inner-join construction row still spends most sampled time in bulk memory movement, with visible Rust-side attribution in `fp_join::build_dense_i64_inner_output_data` and a separate `Column` drop/materialization lifecycle tail through `__munmap`. `inner_join_read` is intentionally much slower than construction-only `inner_join`, which gives the next pass a clear semantic split: avoid right-lane materialization and drop work when the output is only constructed, while preserving deterministic materialization and byte-identical results when consumers force the column contents.
