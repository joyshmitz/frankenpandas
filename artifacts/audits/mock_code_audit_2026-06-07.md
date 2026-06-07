# Mock/Stub/Placeholder Audit — 2026-06-07

Multi-method sweep of all 12 library crates (cc-pandas, /mock-code-finder method).

## Methods run

| Method | Pattern | Result |
|--------|---------|--------|
| Unimplemented macros | `unimplemented!`, `todo!()` | **0 hits** (whole repo) |
| Keyword markers | `TODO\|FIXME\|HACK\|XXX\|STUB\|PLACEHOLDER\|MOCK\|DUMMY` | **0 hits** (whole repo, --type rust) |
| Fake work | `thread::sleep` / `.sleep(` outside tests | **0 hits** |
| Always-true/false bools | short `-> bool` bodies in fp-frame/fp-columnar/fp-index | **0 suspicious** |
| Hardcoded empty returns | `return (Ok())?Vec::new()` / `Ok(String::new())` in fp-io/fp-frame/fp-groupby | 4 hits, **all legitimate edge-case early returns** (empty schema → no PKs, no modes when max_count==0, empty index in asfreq, n==0 in select_extreme_positions) |

## Verdict

**Zero stubs, mocks, or placeholder implementations found in library code.**

Structural reason: the AACE design routes every unsupported pandas surface
through explicit `Err(FrameError::CompatibilityRejected(...))` (758 sites in
fp-frame alone) — unsupported paths fail loud instead of returning fake
results, so placeholder code has nowhere to hide. The conformance corpus +
live-oracle differential testing covers the silent-wrong-results class
separately (see differential-bug-hunting fixes d0d39917 et al.).

## Known silent-divergence inventory (tracked, not stubs)

- `br-frankenpandas-joeff` — value_counts(dropna=False) collapses None/nan/NaT
  into one null bucket (pandas keeps them distinct). Blocked on
  `IndexLabel::Null(NullKind)`; implementation plan + measured blast radius
  (2,678 `IndexLabel::` refs across 6 crates, exhaustive-match-bounded) now
  recorded in the bead.
- `br-frankenpandas-ie5q1` — 5 conformance fixtures drift under pandas 2.2.3
  (legacy-pinned oracle); refresh deferred to the pinned-pandas upgrade.

## Re-audit guidance

Re-run after any large feature merge:
`rg -n "unimplemented!|todo!|TODO|FIXME|STUB|PLACEHOLDER" --type rust`
plus the empty-return and sleep scans above. Expected steady state: 0/0/0.
