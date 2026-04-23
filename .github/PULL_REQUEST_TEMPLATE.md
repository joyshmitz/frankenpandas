<!-- PR template for frankenpandas. Delete this comment before submitting. -->

## Summary

<!-- One-paragraph description of what this PR does + why. -->

## Linked bead

<!-- Required: which bead ID does this PR address? -->
<!-- Format: br-frankenpandas-XXXX (run `br show frankenpandas-XXXX` to read it). -->

Closes: br-frankenpandas-<!-- ID -->

## Scope checklist

<!-- Tick the ones that apply; explain when multiple trigger. -->

- [ ] **Conformance impact**: does this change differential behavior vs pandas?
  - If yes: list the affected live_oracle_* tests OR note "no live-oracle coverage yet; packet fixture added."
- [ ] **Fuzz impact**: does this change a `pub fn` accepting `&[u8]` / `&str` from untrusted sources?
  - If yes: confirm the fuzz target still covers it OR note the new target added.
- [ ] **Performance impact**: does this touch a hot path tracked by `perf_baselines.rs` / a criterion bench?
  - If yes: include before/after p50/p95/p99 numbers.
- [ ] **API stability impact**: does this add a variant to a `#[non_exhaustive]` enum OR change a public signature?
  - If yes: note whether cargo-semver-checks passes; link the release note line (br-4clx cadence).

## Pre-commit diff-stat

<!-- Required by br-iy8u doctrine. Paste the output of `git diff --stat HEAD~1 HEAD` here. -->

```
<!-- git diff --stat output -->
```

If this PR bundles work that also addresses another bead, list the co-landed IDs:

<!-- Co-landed: br-frankenpandas-XXXX, br-frankenpandas-YYYY -->

## Testing

<!-- Required: how did you verify? -->

- [ ] `cargo test --workspace --all-targets` under rch (or locally) — pasted summary:

```
<!-- e.g. "23 suites · 3186 passed · 0 failed · 13 ignored" -->
```

- [ ] `cargo clippy --workspace --all-targets -- -D warnings` — passes with zero warnings.
- [ ] (if touching IO / parser / eval / SQL) corresponding fuzz target short run under rch.
- [ ] (if touching conformance fixtures) `cargo test -p fp-conformance --lib` — 0 regressions.

## DISCREPANCIES.md update?

- [ ] No new intentional pandas divergence in this PR.
- [ ] New DISC-NNN entry added to `crates/fp-conformance/DISCREPANCIES.md`.

## Security

- [ ] No new untrusted-input code paths (if any: SECURITY.md scope stays accurate).
- [ ] `cargo audit` / `cargo deny check` locally green (the CI `security` + `licenses` jobs will re-run).

## Notes for reviewer

<!-- Optional. Anything that needs attention: tricky refactor, known limitation, follow-up bead to file. -->
