# Round 1 Isomorphism Proof

## Change: Eliminate duplicate fixture evaluation in conformance runner

- Location: `crates/fp-conformance/src/lib.rs` (`run_fixture`)
- Lever type: control-flow deduplication (single optimization lever)

Proof obligations:

- Ordering preserved: yes. Fixture iteration order remains sorted by path; only duplicate execution removed.
- Tie-breaking unchanged: yes. Result status still derives from `run_fixture_operation` result for the same fixture data.
- Floating-point behavior: identical. Numeric operations and comparison logic were untouched.
- RNG seeds: unchanged/N/A. No random behavior in this path.
- Golden outputs: `sha256sum -c artifacts/perf/golden_checksums.txt` passed.

Conformance evidence:

- `cargo test -p fp-conformance -- --nocapture` passed.
- `cargo test --workspace` passed.

## Change: Packet-scoped artifact emission and gate drift regression test

- Location: `crates/fp-conformance/src/lib.rs`, `crates/fp-conformance/src/bin/fp-conformance-cli.rs`
- Lever type: conformance observability extension (no execution-kernel behavior change)

Proof obligations:

- Ordering preserved: yes. Fixture sorting and packet grouping remain deterministic (`case_id` order within packet).
- Tie-breaking unchanged: yes. No comparison/equality semantics were altered.
- Floating-point behavior: identical. Numeric kernels and scalar comparison logic unchanged.
- RNG seeds: unchanged/N/A.
- Golden outputs: `sha256sum -c artifacts/perf/golden_checksums.txt` passed after refreshing packet fixtures.

Conformance evidence:

- `cargo run -p fp-conformance --bin fp-conformance-cli -- --oracle fixture --write-artifacts` emitted green reports for `FP-P2C-001` and `FP-P2C-002`.
- `parity_mismatch_corpus.json` emitted for both packets with `mismatch_count = 0`.
- `parity_gate_evaluation_fails_for_injected_drift` test verifies fail path when thresholds are violated.
- Packet family expansion to `FP-P2C-003` remains green under `./scripts/phase2c_gate_check.sh`.
- Packet family expansion to `FP-P2C-004` (`series_join`) remains green under `./scripts/phase2c_gate_check.sh`.

## Change: Blocking gate enforcement mode + drift-history ledger

- Location: `crates/fp-conformance/src/lib.rs`, `crates/fp-conformance/src/bin/fp-conformance-cli.rs`, `scripts/phase2c_gate_check.sh`
- Lever type: release-gate hardening and observability persistence (no compute-kernel semantic change)

Proof obligations:

- Ordering preserved: yes. Report generation order and case sorting are unchanged.
- Tie-breaking unchanged: yes. No change to arithmetic/alignment semantics or comparator behavior.
- Floating-point behavior: identical/N/A for this change set.
- RNG seeds: unchanged/N/A.
- Golden outputs: fixture checksums remain valid in `artifacts/perf/golden_checksums.txt`.

Conformance evidence:

- `cargo test -p fp-conformance -- --nocapture` includes enforcement pass/fail and drift-history tests.
- `./scripts/phase2c_gate_check.sh` now fails closed if packet parity/gate drift appears.
- Drift-history ledger includes packet rows for `FP-P2C-001`, `FP-P2C-002`, `FP-P2C-003`, and `FP-P2C-004`.

## Change: Add `series_join` conformance operation + `FP-P2C-004` packet

- Location: `crates/fp-conformance/src/lib.rs`, `crates/fp-conformance/oracle/pandas_oracle.py`, `crates/fp-conformance/fixtures/packets/fp_p2c_004_*.json`
- Lever type: feature-surface expansion with parity-gated fixtures

Proof obligations:

- Ordering preserved: yes. Join output follows deterministic left-driven traversal with duplicate cross-product expansion.
- Tie-breaking unchanged: yes. For duplicate keys, nested-loop emission order remains stable.
- Floating-point behavior: unchanged for existing operations; join comparison treats missing markers as semantically equivalent when both sides are missing.
- RNG seeds: unchanged/N/A.
- Golden outputs: `sha256sum -c artifacts/perf/golden_checksums.txt` passed after adding packet-004 fixtures.

Conformance evidence:

- `FP-P2C-004` parity report is green with gate pass and zero mismatches.
- `./scripts/phase2c_gate_check.sh` emits green across `FP-P2C-001`..`FP-P2C-004`.

## Change: Add `groupby_sum` conformance operation + `FP-P2C-005` packet

- Location: `crates/fp-conformance/src/lib.rs`, `crates/fp-conformance/oracle/pandas_oracle.py`, `crates/fp-conformance/fixtures/packets/fp_p2c_005_*.json`
- Lever type: feature-surface expansion with packetized parity gate

Proof obligations:

- Ordering preserved: yes. Group output preserves first-seen key order under deterministic scan.
- Tie-breaking unchanged: yes. Repeated keys aggregate in stable iteration order.
- Floating-point behavior: additive behavior unchanged; expected fixtures encode `float64` outputs from aggregation path.
- RNG seeds: unchanged/N/A.
- Golden outputs: `sha256sum -c artifacts/perf/golden_checksums.txt` passed after adding packet-005 fixtures.

Conformance evidence:

- `FP-P2C-005` parity report is green with gate pass and zero mismatches.
- `./scripts/phase2c_gate_check.sh` emits green across `FP-P2C-001`..`FP-P2C-005`.
- `cargo test -p fp-conformance -- --nocapture` passes packet filter and grouped-report tests including `FP-P2C-005`.
