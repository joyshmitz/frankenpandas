# FrankenPandas Execution Tracker

Status legend:
- `[ ]` not started
- `[-]` in progress
- `[x]` completed
- `[!]` blocked/risk

## A. Execution Control

- [x] A1. Create a single granular tracker and keep it updated throughout execution.
- [x] A2. Keep task decomposition synchronized with discovered subtasks during implementation.
- [x] A3. Ensure every completed task has validation evidence recorded.

## B. FP-P2C-002 Dedicated Fixture Corpus + Packet-Specific Reporting

### B1. Fixture schema and operations
- [x] B1.1. Review current `fp-conformance` fixture schema and operation enum.
- [x] B1.2. Add explicit operation(s) for packet `FP-P2C-002` index semantics.
- [x] B1.3. Implement evaluator logic for new operation(s).
- [x] B1.4. Add mismatch formatting for operation-specific diagnostics.

### B2. Dedicated packet fixtures
- [x] B2.1. Add `FP-P2C-002` fixture for deterministic union alignment order.
- [x] B2.2. Add `FP-P2C-002` fixture for duplicate detection behavior.
- [x] B2.3. Add `FP-P2C-002` fixture for first-occurrence position map semantics.
- [x] B2.4. Add strict-mode and hardened-mode variants when behavior differs.
- [x] B2.5. Update fixture manifest under `artifacts/phase2c/FP-P2C-002/fixture_manifest.json`.

### B3. Packet-specific parity generation
- [x] B3.1. Add packet filtering API in `fp-conformance` (`run_packet` by `packet_id`).
- [x] B3.2. Add grouped suite API (`run_packets_grouped`) to produce one report per packet.
- [x] B3.3. Add per-packet pass/fail aggregate struct with strict/hardened counters.
- [x] B3.4. Add tests that validate packet-specific report partitioning.
- [x] B3.5. Replace `FP-P2C-002` proxy parity note with real packet result.

### B4. Gate checks
- [x] B4.1. Parse `parity_gate.yaml` for each packet.
- [x] B4.2. Validate report against gate thresholds.
- [x] B4.3. Emit machine-readable gate result artifact per packet.
- [x] B4.4. Add regression test for gate pass and gate fail examples.

## C. Real RaptorQ Sidecars + Decode Proof Verification

### C1. RaptorQ crate integration
- [x] C1.1. Choose and add a real RaptorQ Rust dependency with explicit version.
- [x] C1.2. Implement symbol generation for parity report payloads.
- [x] C1.3. Persist repair symbol sidecar manifest with OTI/parameters.
- [x] C1.4. Hash and record symbol digests in sidecar metadata.

### C2. Scrub and recovery
- [x] C2.1. Implement integrity scrub that verifies source hash and symbol consistency.
- [x] C2.2. Implement decode-recovery drill by dropping part of source symbols.
- [x] C2.3. Reconstruct payload from mixed source+repair symbols.
- [x] C2.4. Verify recovered payload hash equals source hash.
- [x] C2.5. Emit decode proof artifact with recovery event details.

### C3. Artifact wiring
- [x] C3.1. Replace placeholder `parity_report.raptorq.json` for `FP-P2C-001`.
- [x] C3.2. Replace placeholder `parity_report.raptorq.json` for `FP-P2C-002`.
- [x] C3.3. Replace placeholder decode proof artifacts with real proof entries.
- [x] C3.4. Add test coverage for sidecar generation + decode proof verification.

## D. Direct Legacy pandas Oracle Capture Path

### D1. Oracle runner design
- [x] D1.1. Define operation contract for oracle capture (starting with series add + index operations).
- [x] D1.2. Implement python runner invocation from Rust (`std::process::Command`).
- [x] D1.3. Add environment wiring for legacy tree (`legacy_pandas_code/pandas`) and fallback policy.
- [x] D1.4. Add strict fail-closed behavior when oracle unavailable in strict mode.

### D2. Oracle scripts
- [x] D2.1. Add oracle script under project control (deterministic JSON output).
- [x] D2.2. Script path for `series_add` with index alignment + null behavior.
- [x] D2.3. Script path for index alignment semantics (`FP-P2C-002` ops).
- [x] D2.4. Include explicit error surface mapping from python exceptions.

### D3. Conformance integration
- [x] D3.1. Add mode to conformance runner: static expected vs live oracle capture.
- [x] D3.2. Normalize oracle output into fixture-equivalent shape.
- [x] D3.3. Compare target outputs against live oracle outputs.
- [x] D3.4. Emit mismatch corpus with oracle/target payloads.
- [x] D3.5. Add tests for command wiring and normalization logic.

## E. Performance + Proof Artifacts (expanded)

### E1. Baseline/profile rerun for new pipeline
- [x] E1.1. Re-run `hyperfine` on packet suite command after new features.
- [x] E1.2. Re-run syscall profile (`strace -c`) for new pipeline.
- [x] E1.3. Update `artifacts/perf/ROUND1_BASELINE.md` with new measurements.

### E2. Isomorphism + golden outputs
- [x] E2.1. Refresh golden output bundle if fixture set changed.
- [x] E2.2. Recompute `golden_checksums.txt`.
- [x] E2.3. Re-run checksum verification.
- [x] E2.4. Update `ROUND1_ISOMORPHISM_PROOF.md` with additional change levers.

## F. Documentation + Parity Artifacts

### F1. Packet artifacts
- [x] F1.1. Update `artifacts/phase2c/FP-P2C-001/parity_report.json` from new run.
- [x] F1.2. Update `artifacts/phase2c/FP-P2C-002/parity_report.json` from new run.
- [x] F1.3. Ensure `risk_note.md` reflects new behavior and residual drift.
- [x] F1.4. Ensure `fixture_manifest.json` entries are exact and complete.

### F2. Project docs
- [x] F2.1. Update `FEATURE_PARITY.md` statuses/notes based on completed work.
- [x] F2.2. Update `README.md` with oracle and sidecar capabilities now implemented.
- [x] F2.3. Update `PROPOSED_ARCHITECTURE.md` for packet/gate/oracle/sidecar flows.

## G. Full Validation + Completion

- [x] G1. Run `cargo fmt --check`.
- [x] G2. Run `cargo check --all-targets`.
- [x] G3. Run `cargo clippy --all-targets -- -D warnings`.
- [x] G4. Run `cargo test --workspace`.
- [x] G5. Run `cargo test -p fp-conformance -- --nocapture`.
- [x] G6. Run `cargo bench`.
- [x] G7. Record all command outcomes in final report with failures/fallbacks, if any.

## H. Risks/Blocks Tracking

- [x] H1. Monitor `/tmp` full condition impact and use safe workarounds.
- [x] H2. Confirm no destructive operations are used.
- [x] H3. Track any oracle runtime dependency gaps (python/pandas import issues).

## I. Blocking Gate Automation + Drift Ledger (Follow-On)

- [x] I1. Add fail-closed gate enforcement API in `fp-conformance` for grouped reports.
- [x] I2. Add append-only packet drift history writer (`artifacts/phase2c/drift_history.jsonl`).
- [x] I3. Add CLI flags to enable gate enforcement and explicit drift-history writes.
- [x] I4. Add unit tests for enforcement pass/fail and drift-history row emission.
- [x] I5. Add operational gate-check script (`scripts/phase2c_gate_check.sh`).
- [x] I6. Add CI workflow to run required cargo checks and phase2c gate script.
- [x] I7. Update docs/spec/parity references for new gating/drift-history path.

## J. Packet Expansion FP-P2C-003 (Follow-On)

- [x] J1. Add three new `FP-P2C-003` fixtures for strict/hardened series-add semantics.
- [x] J2. Add packet metadata docs (`contract_table.md`, `legacy_anchor_map.md`, `risk_note.md`).
- [x] J3. Add `fixture_manifest.json` and `parity_gate.yaml` for `FP-P2C-003`.
- [x] J4. Run gate-check script and generate full artifact set for `FP-P2C-003`.
- [x] J5. Update parity/spec/readme docs to include `FP-P2C-003`.
- [x] J6. Refresh golden output checksums to include packet-003 fixtures.

## K. Packet Expansion FP-P2C-004 Join Semantics (Follow-On)

- [x] K1. Add new fixture operation support for `series_join` with explicit `join_type`.
- [x] K2. Add join expected schema and evaluator path in conformance harness.
- [x] K3. Add live oracle adapter support for `series_join`.
- [x] K4. Add `FP-P2C-004` join fixtures and packet metadata/gate contracts.
- [x] K5. Regenerate artifacts and validate gate green for `FP-P2C-004`.
- [x] K6. Refresh golden checksums and parity/spec docs for packet-004 coverage.

## L. Packet Expansion FP-P2C-005 GroupBy Sum Semantics (Current)

- [x] L1. Add conformance operation wiring for `groupby_sum` in harness + oracle adapter.
- [x] L2. Add `FP-P2C-005` fixture corpus (strict ordering, strict alignment/dropna, hardened int keys).
- [x] L3. Add packet metadata docs (`fixture_manifest.json`, `parity_gate.yaml`, `contract_table.md`, `legacy_anchor_map.md`, `risk_note.md`).
- [x] L4. Regenerate packet artifacts and validate gate green for `FP-P2C-005`.
- [x] L5. Refresh golden checksums and performance/isomorphism evidence for packet-005 coverage.
- [x] L6. Update parity/spec/readme tracker references and run full required validation stack.

## M. Phase-2C Packet Roadmap (Planned, Granular Backlog)

- [ ] M1. Define and reserve packet IDs for next parity slices (`FP-P2C-006+`) with explicit scope boundaries.
- [ ] M2. Add `FP-P2C-006` packet for groupby aggregate matrix (`sum/mean/count`, null-heavy and skewed keys).
- [ ] M3. Add `FP-P2C-007` packet for filter/mask semantics (`loc` boolean masks, null-mask behavior).
- [ ] M4. Add `FP-P2C-008` packet for CSV malformed-ingest parity (strict fail-closed vs hardened bounded recovery).
- [ ] M5. Add `FP-P2C-009` packet for dtype promotion/coercion matrix parity.
- [ ] M6. Add `FP-P2C-010` packet for deterministic output ordering contracts across join/groupby chains.
- [ ] M7. Add packet-level adversarial fixtures and fuzz harness seeds for each new packet family.
- [ ] M8. Extend packet gate policy taxonomy to include explicit divergence categories per packet.

## N. asupersync + frankentui Deep Integration (Planned)

- [ ] N1. Extend `fp-runtime` outcome bridge to carry packet gate summaries and mismatch corpus pointers.
- [ ] N2. Add deterministic asupersync sync schema for conformance/perf artifact bundles.
- [ ] N3. Implement FTUI packet dashboard cards (gate state, drift trend, decode-proof status).
- [ ] N4. Add FTUI drilldown views for mismatch corpus replay and evidence ledger traces.
- [ ] N5. Add strict/hardened mode toggle telemetry surfaces in FTUI with explicit policy provenance.
- [ ] N6. Add integration tests that validate asupersync + FTUI contract compatibility under packet drift.

## O. Full Port Completion and Hardening (Planned)

- [ ] O1. Build comprehensive pandas API conformance matrix and map every scoped API to packet families.
- [ ] O2. Implement missing constructor/indexing/join/groupby/IO functionality to reach scoped 100% parity.
- [ ] O3. Add differential harness expansion for live-oracle replay across all packet families and API matrix rows.
- [ ] O4. Add benchmark suites for filter/groupby/join kernels with p50/p95/p99 and memory/allocation budgets.
- [ ] O5. Add compatibility drift gates in CI for both fixture and live oracle modes.
- [ ] O6. Add threat-model and adversarial test docs per major subsystem (ingest, coercion, state transitions).
- [ ] O7. Add release-grade artifact manifests with RaptorQ sidecars for conformance + benchmark bundles.
- [ ] O8. Add reproducibility ledger (`env`, `manifest`, lockfiles) for deterministic reruns.

## Evidence Ledger (Session)

- Validation commands passed:
  - `cargo fmt --check`
  - `cargo check --all-targets`
  - `cargo clippy --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo test -p fp-conformance -- --nocapture`
  - `cargo bench`
  - `./scripts/phase2c_gate_check.sh`
- Fail-closed behavior check:
  - `cargo run -p fp-conformance --bin fp-conformance-cli -- --packet-id FP-P2C-001 --oracle live --require-green` exits non-zero with gate drift reasons.
- Conformance packet artifacts regenerated:
  - `artifacts/phase2c/FP-P2C-001/*`
  - `artifacts/phase2c/FP-P2C-002/*`
  - `artifacts/phase2c/FP-P2C-003/*`
  - `artifacts/phase2c/FP-P2C-004/*`
  - `artifacts/phase2c/FP-P2C-005/*`
- Blocking gate automation artifacts:
  - `scripts/phase2c_gate_check.sh`
  - `.github/workflows/ci.yml`
  - `artifacts/phase2c/drift_history.jsonl`
- Performance/proof artifacts refreshed:
  - `artifacts/perf/ROUND1_BASELINE.md`
  - `artifacts/perf/round1_packet_hyperfine.json`
  - `artifacts/perf/round1_packet_strace.txt`
  - `artifacts/perf/golden_checksums.txt`
  - `artifacts/perf/ROUND1_ISOMORPHISM_PROOF.md`
