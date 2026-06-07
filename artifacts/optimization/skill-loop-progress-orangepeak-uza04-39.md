# br-frankenpandas-uza04.39 optimization ledger

Agent: OrangePeak
Date: 2026-06-07
Target: dense Int64 inner join, high-fanout right lanes

## Profile-backed target

Baseline/profiling came from the `br-frankenpandas-uza04.38` run on RCH worker
`ts1`:

- `inner_join 100000 3`: 121.3 ms +/- 5.2 ms in the first baseline,
  then 129.2 ms +/- 5.2 ms in the `.39` paired baseline.
- `inner_join_read 100000 1`: 146.3 ms +/- 10.6 ms first baseline,
  then 221.2 ms +/- 36.3 ms in the `.39` paired baseline.
- `perf report` for `inner_join 100000 3` showed the residual dominated by
  eager materialization/memory movement: `__memmove_avx_unaligned_erms`
  78.13% children, `drop_in_place::<fp_columnar::Column>` 12.85%,
  `__munmap` 11.81%.

The `.38` fp-join-only serial right-lane replay preserved goldens but regressed
`inner_join 100000 3` from 121.3 ms to 384.2 ms, so that source hunk was
removed and the next primitive moved across the fp-columnar/fp-join boundary.

## Alien primitive

Chosen primitive: columnar zero-copy/chunked/run-container representation.

Instead of eagerly concatenating the dense-join right lane into a contiguous
output Vec, the right lane now carries:

- one bucket-order `Vec<i64>` value tape;
- ordered `(start, len)` segment descriptors matching left probe/output order;
- lazy `OnceLock` expansion only when `as_i64_slice()` or scalar values are
  actually requested.

This is pure safe Rust and does not link BLAS/LAPACK/MKL/XLA or any C kernel.

## One lever

Implemented exactly one retained performance lever:

- Added `ScalarValues::LazyRepeatedSlicesInt64` and
  `Column::from_i64_repeated_slices` in `fp-columnar`.
- Wired dense high-fanout inner-join right lanes in `fp-join` to emit repeated
  slices when left lanes already use repeat runs.
- Kept the existing eager path for low-fanout/1:1 joins and all non-Int64,
  null/mixed, sorted, indicator, validate, and non-inner fallbacks.
- Expansion is deterministic and safe: row order is the original left probe
  order; right duplicate tie order is the existing bucket insertion order via
  `plan.positions`; final materialization copies exactly the same slices that
  the old eager path copied.

## Isomorphism proof

Golden-output SHA-256 verification:

- `inner_join 100000`
  - before: `494106fca6e3310a318f1685c74041a2788089a4d2409107d4eef4a00c7a0764`
  - final after: `494106fca6e3310a318f1685c74041a2788089a4d2409107d4eef4a00c7a0764`
- `join_1to1 100000`
  - before: `102690aa39952cc2d13bcc41547aacdeac1946113e43d62472fdb93440bc56a7`
  - final after: `102690aa39952cc2d13bcc41547aacdeac1946113e43d62472fdb93440bc56a7`

Byte comparisons passed for both final-after outputs against their before
goldens. The scenario has no RNG or floating-point arithmetic in the changed
lane construction; dtype/name/column order and duplicate ordering are included
in the perf-profile golden serialization.

## Same-worker benchmark

All retained numbers below are on worker `ts1`.

| Scenario | Before | Final after | Delta |
| --- | ---: | ---: | ---: |
| `inner_join 100000 3` | 129.2 ms +/- 5.2 | 45.5 ms +/- 1.7 | 2.84x faster |
| `join_1to1 100000 20` | 40.3 ms +/- 3.8 | 37.4 ms +/- 2.3 | 1.08x faster/control clean |
| `inner_join_read 100000 1` | 221.2 ms +/- 36.3 | 149.7 ms +/- 13.2 | 1.48x faster |

Kept score: Impact 5 x Confidence 4 / Effort 2 = 10.0.

## Validation

- `rch exec -- cargo test -p fp-columnar repeated_slice_int64_column_matches_eager_materialization --lib`: pass.
- `rch exec -- cargo test -p fp-join dense_i64_inner_high_fanout_emits_repeat_run_left_lanes --lib`: pass.
- `rch exec -- cargo test -p fp-columnar --lib`: 374 passed, 5 ignored.
- `rch exec -- cargo test -p fp-join --lib`: 102 passed.
- `rch exec -- cargo clippy -p fp-columnar --all-targets -- -D warnings`: pass after package example lint cleanup.
- `rch exec -- cargo clippy -p fp-join --all-targets -- -D warnings`: pass.
- `cargo fmt --check -p fp-columnar -p fp-join`: pass.
- Full workspace `cargo fmt --check`: not used as a gate for this crate-scoped
  campaign because it reports unrelated pre-existing formatting diffs in other
  crates.
- `ubs $(git diff --name-only -- '*.rs')`: non-zero due false-positive security
  heuristics on `fp-join` dtype/key equality plus broad existing warning
  inventory; UBS's internal `cargo check`, clippy, formatting, and test-build
  checks were clean.

## Artifacts

- `tests/artifacts/perf/fp_before_uza04_39_orangepeak.json`
- `tests/artifacts/perf/fp_after_uza04_39_final20_orangepeak.json`
- `tests/artifacts/perf/golden_before_uza04_39_inner_join_orangepeak.sha256`
- `tests/artifacts/perf/golden_after_uza04_39_final_inner_join_orangepeak.sha256`
- `tests/artifacts/perf/golden_before_uza04_39_join_1to1_orangepeak.sha256`
- `tests/artifacts/perf/golden_after_uza04_39_final_join_1to1_orangepeak.sha256`

The raw inner-join golden text files were generated and byte-compared locally,
but are not intended to be committed because they are about 932 MiB each. The
committed SHA files are the compact golden proof.
