# Skill Loop Progress
# Skill: extreme-software-optimization + alien-graveyard + alien-artifact-coding
# Target: br-frankenpandas-uza04.38 / dense inner-join output column lifecycle
# Total Passes: 1
# Started: 2026-06-07T03:53:04Z

## Status: REJECTED - 1 of 1

## Mission
1. Baseline, profile, trial one fp-join-only right-lane materialization lever,
   prove behavior unchanged, keep only if Score >= 2.0.

## Completed Passes
### Pass 1 - Right-Lane Serial Tape Replay Rejection - 2026-06-07T04:03Z
- Bead: `br-frankenpandas-uza04.38`.
- Profile-backed target: `perf_before_uza04_38_inner_join_orangepeak.data`
  showed dense inner join dominated by `__memmove_avx_unaligned_erms`
  (78.13% children), `drop_in_place::<fp_columnar::Column>` (12.85%),
  and `__munmap` (11.81%).
- Baseline, remote `ts1`: `inner_join 100000 3` 121.3 ms +/- 5.2 ms;
  `join_1to1 100000 20` 35.0 ms +/- 1.5 ms; `inner_join_read 100000 1`
  146.3 ms +/- 10.6 ms.
- Trial lever: when high fanout made all left lanes lazy repeat-runs and only
  right lanes were full, replay right bucket value tapes with
  `Vec::with_capacity` plus `extend_from_slice`, avoiding the existing
  parallel path's pre-zeroed `vec![0; output_len]` buffer.
- Behavior proof:
  - Ordering: left probe order and right bucket insertion order were unchanged.
  - Tie-breaking: duplicate-key output order stayed left row then right bucket
    order.
  - Floating point: N/A; dense lane is Int64-only.
  - RNG/hash behavior: unchanged; no randomized path introduced.
  - Golden sha256 stayed byte-identical:
    - `inner_join`: `494106fca6e3310a318f1685c74041a2788089a4d2409107d4eef4a00c7a0764`
    - `join_1to1`: `102690aa39952cc2d13bcc41547aacdeac1946113e43d62472fdb93440bc56a7`
- After, remote `ts1`: `inner_join 100000 3` 384.2 ms +/- 32.1 ms;
  `join_1to1 100000 20` 41.2 ms +/- 5.1 ms; `inner_join_read 100000 1`
  227.0 ms +/- 21.3 ms.
- Verdict: REJECTED. The serial one-write path lost to the existing parallel
  pre-zero plus overwrite path. Source hunk removed.
- Next route: avoid another fp-join loop-order tweak. The deeper primitive is
  a safe-Rust lazy/chunked right-lane column representation (repeat-slice /
  dictionary-run / chunked array) so the output can avoid eager contiguous
  materialization without moving the cost into a serial replay.
