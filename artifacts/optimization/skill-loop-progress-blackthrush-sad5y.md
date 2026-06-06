# Skill Loop Progress
# Skill: extreme-software-optimization
# Target: br-frankenpandas-sad5y and subsequent unowned perf hotspots
# Total Passes: 6
# Started: 2026-06-06T00:30:00Z

## Status: IN PROGRESS - Pass 3 of 6

## Coordination Note
Root `.skill-loop-progress.md` is exclusively reserved by RubyGoose, so this
bead-scoped progress file is the active BlackThrush loop ledger until that lease
is released.

## Missions
1. Kendall bounded-rank inversion counter: replace the profiled recursive usize
   inversion merge in the complete/no-tie Kendall matrix path with one
   bounded-rank Fenwick counter, preserving exact discordant-pair arithmetic.
2. Kendall post-keep reprofile: after pass 1 lands or rejects, reprofile
   `df_kendall` and route the next measured residual without touching
   RubyGoose-owned alignment/storage lanes.
3. Alien primitive harvest: match the top current hotspot to the canonical
   graveyard docs and choose one EV>=2.0 structural primitive, not a nearby
   micro-tweak.
4. Artifact proof hardening: produce the isomorphism ledger for ordering,
   tie-breaking, floating-point, RNG, and golden sha256 across the active lane.
5. Next unowned perf bead pass: claim the top ready or profiler-evident perf
   bead, baseline by RCH, and ship or reject one lever by score.
6. Campaign closeout and reroute: close completed beads, sync artifacts, push
   main and main:master, and leave the next profile-backed target explicit.

## Completed Passes
1. `br-frankenpandas-sad5y`: kept bounded-rank Fenwick inversion counting for
   no-tie Kendall matrix correlation. Hyperfine mean improved from 62.4 ms +/-
   6.5 ms to 41.0 ms +/- 1.8 ms on `df_kendall 512 20`, a 1.5213x speedup
   and 34.26% mean reduction. Golden output sha256 stayed
   `dcce8a4e1f13d887361650ec24fcfee13c6c4c3954936fd124a1feb53cf125ce`.
2. `br-frankenpandas-lxnar`: rejected packed row-pair bitsets for complete
   no-tie Kendall matrices. Golden output stayed identical, but hyperfine
   regressed from 41.8 ms +/- 1.7 ms to 143.0 ms +/- 5.5 ms on
   `df_kendall 512 20`; the source hunk was removed.

## Pass 1 Evidence - Pre-Edit
- Bead: br-frankenpandas-sad5y.
- Profile: `perf_df_kendall_fenwick_before_blackthrush_sad5y.data` shows
  `<fp_frame::Series>::count_usize_inversions_recursive` at about 69.6% self
  samples for `df_kendall`.
- Refreshed RCH build: `cargo build -p fp-conformance --profile release-perf
  --example perf_profile`, worker `ts1`, exit 0.
- Refreshed baseline: `df_kendall 512 20`, hyperfine mean 62.4 ms +/- 6.5 ms.
- Golden before sha256:
  `dcce8a4e1f13d887361650ec24fcfee13c6c4c3954936fd124a1feb53cf125ce`.
- Alien primitive match: bounded prefix-count/rank structure. Canonical
  graveyard docs identify rank/select structures as the integer-query substrate;
  the FrankenSuite summary identifies Fenwick trees as an O(log n) primitive.
- Opportunity score: Impact 4 x Confidence 4 / Effort 2 = 8.0.
- Fallback trigger: if golden sha256 changes, focused Kendall tests fail, or
  after hyperfine does not clear Score >= 2.0, reject/revert this lever.

## Pass 1 Evidence - Post-Edit
- Lever: replaced allocation plus recursive merge inversion counting over
  `y_order` with a bounded Fenwick counter over `y_rank_by_row[x_order]`.
- Behavior proof:
  - Ordering: `x_order` traversal is unchanged; only the discordant-pair counter
    changes representation.
  - Tie-breaking: fast path still declines ties before counting, so tie semantics
    remain on the existing fallback path.
  - Floating point: discordant pair count remains exact integer arithmetic; final
    tau formula is unchanged.
  - RNG: benchmark/oracle input generation is unchanged.
  - Golden sha256 before and after:
    `dcce8a4e1f13d887361650ec24fcfee13c6c4c3954936fd124a1feb53cf125ce`.
- Refreshed after RCH build: `cargo build -p fp-conformance --profile
  release-perf --example perf_profile`, worker `ts1`, exit 0.
- Focused behavior tests: `cargo test -p fp-frame --lib kendall -- --nocapture`,
  worker `ts1`, 5 tests passed.
- Compile gates: `cargo check -p fp-frame --all-targets` and `cargo clippy -p
  fp-frame --all-targets -- -D warnings` passed via RCH.
- `cargo fmt --check -p fp-frame` is blocked by pre-existing broad fp-frame
  formatting drift, so no repository-wide formatting rewrite was performed.
- UBS on `crates/fp-frame/src/lib.rs` was inconclusive because the scanner stayed
  in an `ast-grep` unwrap pass for several minutes; only that scanner process
  group was terminated.
- After profile artifact: `perf_df_kendall_fenwick_after_blackthrush_sad5y.data`.
  The old recursive inversion symbol is absent from the visible hot list. Current
  residuals include the parallel Kendall matrix closure, `pairwise_rank_corr`
  column-name iteration, and `kendall_no_tie_order` sorting.
- Keep score: Impact 4 x Confidence 4 / Effort 2 = 8.0.

## Pass 2 Target - Post-Keep Reprofile
- Profile-backed residual: `df_kendall 512 20` now shows sorting/order
  construction and pairwise matrix orchestration overhead after the inversion
  counter keep.
- Required next step: claim or create the next unowned perf bead for a measured
  Kendall residual, then baseline one structural lever before editing.

## Pass 2 Evidence - Rejected Bit-Parallel Matrix
- Bead: `br-frankenpandas-lxnar`.
- Lever attempted: encode each complete no-tie column's row-pair ordering as
  packed `u64` words, then compute discordant counts via XOR + popcount.
- Graveyard/artifact match: succinct bitvectors/rank-select and bitwise
  Hamming-distance primitives. The match was structurally different from the
  Fenwick path, but the constants were wrong for the 512-row / 32-column
  benchmark shape.
- Behavior proof:
  - Ordering: row-pair iteration used deterministic `(left, right)` row order.
  - Tie-breaking: only the complete no-tie fast path was eligible; existing
    tie/non-finite fallback remained.
  - Floating point: final tau formula and integer discordant counts stayed
    exact.
  - RNG: unchanged / N/A.
  - Golden sha256 before and after:
    `dcce8a4e1f13d887361650ec24fcfee13c6c4c3954936fd124a1feb53cf125ce`.
- RCH baseline: `df_kendall 512 20`, 41.8 ms +/- 1.7 ms.
- RCH after: `df_kendall 512 20`, 143.0 ms +/- 5.5 ms.
- Verdict: REJECTED. Score below threshold because the O(cols * rows^2) bitset
  construction dominated this workload. Source hunk removed.

## Pass 3 Target - Fenwick Scratch / Orchestration Residual
- Profile-backed residual remains the complete no-tie Kendall matrix closure.
- Next primitive: reuse per-worker Fenwick scratch/epoch state or otherwise
  remove per-pair allocation/zeroing in the matrix closure, while keeping the
  already-proven exact discordant-pair arithmetic.
- Fallback trigger: reject if same-harness hyperfine does not beat the 41.8 ms
  baseline by Score >= 2.0 or if golden sha256 changes.

## Pass 3 Result - Fenwick Scratch Reuse Rejection
- Bead: `br-frankenpandas-izmsk`.
- Trial lever: reused one worker-local Fenwick `ranks_seen` buffer inside
  `complete_kendall_no_tie_parallel_matrix` and threaded it into the ordered
  rank inversion helper.
- RCH candidate build: `CARGO_TARGET_DIR=/data/projects/.scratch/cargo-target-blackthrush-izmsk-after rch exec -- cargo build -p fp-conformance --profile release-perf --example perf_profile`, worker `ts1`, exit 0.
- Saved before baseline: `df_kendall 512 20`, 41.1 ms +/- 1.7 ms.
- Paired A/B: before 43.2 ms +/- 2.6 ms, candidate 45.6 ms +/- 1.9 ms,
  so the before binary was 1.05x +/- 0.08 faster.
- Golden correction: the originally saved before golden was stale. Rerunning
  the saved before binary produced
  `46a26659e27d862c5de8a9b030c422028e42f5f58c0d977eb7153898f4a5818d`,
  matching the candidate byte-for-byte.
- Isomorphism: row/column ordering, tie/non-finite fallback, integer
  discordant-pair arithmetic, final tau formula, and RNG behavior were
  unchanged in the trial, but the performance gate failed.
- Verdict: REJECTED. Source hunk removed; artifacts retained for audit.
