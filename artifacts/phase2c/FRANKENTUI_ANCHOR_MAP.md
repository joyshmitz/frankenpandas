# FRANKENTUI Anchor Map + Behavior/Workflow Extraction Ledger

Bead: `bd-2gi.28.1` [FRANKENTUI-A]
Subsystem: FRANKENTUI -- terminal user interface for the FrankenPandas operator cockpit (conformance, performance, and forensics dashboards)

## Legacy Anchors

### External Dependency (Planned)

- `frankentui` crate -- TUI framework at `/dp/frankentui` (per FrankenSQLite spec, section 1.3)
- FrankenSQLite reference usage: `fsqlite-cli` crate depends on `frankentui` for interactive shell (dot-commands, output modes, tab completion, syntax highlighting, history)
- Reference spec locations:
  - `references/frankensqlite/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md` line 182 -- dependency table entry
  - `references/frankensqlite/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md` line 11967 -- crate tree (`fsqlite-cli` uses frankentui)
  - `references/frankensqlite/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md` line 12306 -- fsqlite-cli description (interactive shell using frankentui)
  - `references/frankensqlite/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md` line 12360 -- dependency edge (`fsqlite-cli -> frankentui`)

### FrankenPandas Integration Points (Existing)

- `README.md` line 32 -- "FTUI-ready galaxy-brain decision cards for transparency surfaces"
- `PROPOSED_ARCHITECTURE.md` line 31 -- "decision records can be rendered as FTUI-friendly galaxy-brain cards"
- `TODO_EXECUTION_TRACKER.md` section N (lines 175-182) -- planned deep integration items N1-N6
- `crates/fp-runtime/src/lib.rs` lines 89-121 -- `GalaxyBrainCard` struct and `decision_to_card()` function (the only implemented FTUI-facing surface)
- `crates/fp-runtime/src/lib.rs` lines 97-105 -- `GalaxyBrainCard::render_plain()` produces a 4-line text rendering
- `crates/fp-runtime/src/lib.rs` line 601 -- test: `decision_card_is_renderable_for_ftui_consumers()`
- `crates/fp-frankentui/src/lib.rs` -- new FRANKENTUI foundation crate with read-only artifact ingestion and dashboard/app-state skeleton types
- `crates/fp-frankentui/src/lib.rs` -- `FtuiDataSource` + `FsFtuiDataSource` for deterministic packet discovery and artifact loading
- `crates/fp-frankentui/src/lib.rs` -- `ConformanceDashboardSnapshot`, `ForensicLogSnapshot`, `GovernanceGateSnapshot`, and `FtuiAppState` for panel-layer scaffolding

### Data Sources FTUI Must Consume

- `crates/fp-conformance/src/lib.rs` -- all conformance, forensic, and CI pipeline types:
  - `PacketParityReport` (lines 370-385) -- per-packet pass/fail with fixture counts
  - `PacketGateResult` (lines 388-397) -- gate evaluation with strict/hardened failure counts
  - `PacketDriftHistoryEntry` (lines 440-451) -- drift history JSONL rows
  - `DifferentialReport` (lines 363-367) -- drift-classified differential results
  - `DriftRecord` (lines 292-297) -- individual drift observation (category, level, location, message)
  - `DriftSummary` (lines 353-360) -- drift distribution by category
  - `ForensicLog` (lines 3289-3325) -- timestamped forensic event accumulator
  - `ForensicEventKind` (lines 3226-3278) -- 10-variant event taxonomy (SuiteStart, SuiteEnd, PacketStart, PacketEnd, CaseStart, CaseEnd, ArtifactWritten, GateEvaluated, DriftHistoryAppended, Error)
  - `FailureForensicsReport` (lines 3626-3640) -- human-readable failure digest summary
  - `FailureDigest` (lines 3595-3603) -- per-case failure with replay command
  - `ArtifactId` (lines 3562-3591) -- deterministic artifact cross-reference (SHA-256 short hash)
  - `CiPipelineResult` (lines 1488-1535) -- gate pipeline display with PASS/FAIL per gate
  - `CiGateResult` (lines 1478-1484) -- individual gate result (gate, passed, elapsed_ms, summary, errors)
  - `CiGate` (lines 1376-1468) -- 9-gate enum (G1Compile through G8E2e) with labels and ordering
  - `E2eReport` (lines 3368-3386) -- full E2E orchestration result with forensic log
  - `RaptorQSidecarArtifact` (lines 418-427) -- scrub report and packet records
  - `DecodeProofArtifact` (lines 1163-1170) -- decode proof with status enum
  - `DecodeProofStatus` (lines 1172-1189) -- Recovered/Failed/NotAttempted
- `crates/fp-runtime/src/lib.rs` -- runtime decision types:
  - `DecisionRecord` (lines 78-87) -- Bayesian decision with timestamp, mode, action, issue, metrics, evidence
  - `GalaxyBrainCard` (lines 90-95) -- title, equation, substitution, intuition (all String)
  - `EvidenceLedger` (lines 123-143) -- append-only decision record sequence
  - `ConformalPredictionSet` (lines 393-404) -- conformal guard output
  - `ConformalGuard` (lines 408-551) -- rolling calibration window with coverage tracking
  - `RuntimePolicy` (lines 147-256) -- strict/hardened mode with Bayesian decision dispatch
  - `RaptorQEnvelope` (lines 325-376) -- artifact durability metadata

### Artifact Files FTUI Must Display

- `artifacts/phase2c/{packet_id}/parity_report.json` -- per-packet parity report
- `artifacts/phase2c/{packet_id}/parity_gate_result.json` -- gate evaluation result
- `artifacts/phase2c/{packet_id}/parity_mismatch_corpus.json` -- mismatched fixture details
- `artifacts/phase2c/{packet_id}/parity_report.raptorq.json` -- RaptorQ sidecar
- `artifacts/phase2c/{packet_id}/parity_report.decode_proof.json` -- decode proof artifact
- `artifacts/phase2c/drift_history.jsonl` -- append-only cross-run drift trend JSONL
- `artifacts/schemas/*.schema.json` -- 11 JSON schemas defining artifact structure

### Reference Specifications

- `COMPREHENSIVE_SPEC_FOR_FRANKENPANDAS_V1.md` -- conformance contract (packets, gates, mismatch corpus, drift ledger)
- `artifacts/phase2c/FAILURE_FORENSICS_UX.md` (bd-2gi.21) -- failure digest format, forensics report, artifact index, replay patterns
- `artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md` (bd-2gi.20) -- user workflow corpus (UW-NNN scenarios)
- `artifacts/phase2c/ASUPERSYNC_ANCHOR_MAP.md` (bd-2gi.27.1) -- asupersync integration contract including planned FTUI items N3-N6

---

## Extracted Behavioral Contract

### Normal Conditions

1. **Galaxy-brain card rendering (implemented):** `decision_to_card(&DecisionRecord) -> GalaxyBrainCard` produces a 4-field card with: title (`{subject}::{action:?}`), equation (`argmin_a ...`), substitution (posterior + expected losses), intuition (static text). `render_plain()` concatenates these as `[{title}]\n{equation}\n{substitution}\n{intuition}`. This remains the canonical runtime decision-card primitive consumed by FTUI.

2. **Packet dashboard cards (foundation implemented, N3 rendering pending):** `fp-frankentui` now loads packet parity/gate/decode/mismatch artifacts into `PacketSnapshot` and aggregates packet-level summaries in `ConformanceDashboardSnapshot`. Full interactive TUI cards remain planned.

3. **Drilldown views (planned, N4):** FTUI must provide drill-into views for: mismatch corpus replay (listing `FailureDigest` entries with their replay commands and artifact paths), and evidence ledger traces (listing `DecisionRecord` entries with full Bayesian metrics and evidence terms).

4. **Mode toggle telemetry surfaces (foundation implemented, N5 rendering pending):** `summarize_decision_dashboard()` now emits strict/hardened record counts plus `RuntimePolicy` provenance (`mode`, `fail_closed_unknown_features`, `hardened_join_row_cap`). Interactive mode toggles remain planned.

5. **CI gate pipeline visualization (planned):** FTUI must render the 9-gate CI pipeline (G1Compile through G8E2e) as an ordered pass/fail status bar with per-gate elapsed time and failure summaries, consuming `CiPipelineResult`.

6. **Conformal guard dashboard (planned):** FTUI must surface `ConformalGuard` status: calibration count, empirical coverage rate, coverage alert state, and recent `ConformalPredictionSet` entries showing quantile threshold vs. current score.

7. **Forensic log event stream (foundation implemented, rendering planned):** `load_forensic_log()` now parses forensic JSONL with malformed-line tolerance into `ForensicLogSnapshot`. Full event-stream widgets and filtering UX remain planned.

8. **Drift history trend visualization (foundation implemented, rendering planned):** `load_drift_history()` plus `PacketTrendSnapshot` provide drift-history ingestion and packet-level latest-trend extraction. Sparkline/timeline rendering remains planned.

### Edge Conditions

9. **Empty drift history:** When `drift_history.jsonl` does not exist or is empty, FTUI must display "No drift history available" rather than crashing or rendering an empty chart.

10. **Uncalibrated conformal guard:** When `ConformalGuard::is_calibrated()` returns false (fewer than 2 scores), FTUI must show "Uncalibrated" status and suppress quantile threshold display. The guard returns `quantile_threshold: f64::INFINITY` and admits all actions when uncalibrated.

11. **All-green E2E report:** When `E2eReport::is_green()` returns true (zero failures, at least one fixture, all gates pass), FTUI should render a summary-only view ("ALL GREEN: N/N fixtures passed") per the `FailureForensicsReport::Display` contract.

12. **Clock skew in timestamps:** Decision records may carry `ts_unix_ms = 0` when `SystemTime::now()` fails (the `now_unix_ms()` fallback). FTUI must handle timestamp `0` gracefully (display as "unknown" or epoch, not crash on date formatting).

13. **Placeholder RaptorQ envelopes:** When `RaptorQEnvelope` has `source_hash: "blake3:placeholder"`, `k: 0`, `repair_symbols: 0`, FTUI must display these as "placeholder" status, not as corruption or missing data.

14. **Missing artifact files:** When an artifact path referenced by `WrittenPacketArtifacts` or `FailureDigest` does not exist on disk, FTUI must display "artifact not found" with the expected path, not panic.

15. **Mixed oracle sources:** `DifferentialResult` carries `oracle_source: FixtureOracleSource` (Fixture or Live). FTUI must distinguish fixture-based results from live-oracle results in the drilldown view.

16. **Large forensic logs:** `ForensicLog::events` is an unbounded `Vec<ForensicEvent>`. FTUI must handle logs with thousands of entries via pagination or virtual scrolling, not by loading all entries into a single render pass.

### Adversarial Conditions

17. **Malformed JSONL in drift history:** If `drift_history.jsonl` contains non-JSON lines or truncated entries, FTUI must skip malformed lines and display a warning count, not abort rendering the entire trend.

18. **Concurrent drift history writes:** `drift_history.jsonl` is append-only. If FTUI reads the file while a conformance run is appending, partial last lines may be encountered. FTUI must tolerate trailing incomplete lines.

19. **Adversarial mismatch corpus size:** `parity_mismatch_corpus.json` could contain very large mismatches (e.g., thousands of failing fixtures). FTUI must paginate or truncate the display with "showing N of M mismatches".

20. **Decision engine posterior extremes:** `DecisionMetrics::posterior_compatible` can be extremely close to 0.0 or 1.0 (clamped at 1e-15 boundaries in the conformal score). FTUI must render these without floating-point display artifacts (e.g., showing "0.0000" or "1.0000" rather than scientific notation).

21. **Denial-of-service via evidence terms:** `DecisionRecord::evidence` is a `Vec<EvidenceTerm>` with no length bound. A record with thousands of evidence terms must not freeze the UI. FTUI must cap rendered evidence terms with a "and N more" indicator.

22. **Feature-flag isolation:** FTUI must function with or without the `asupersync` feature enabled in `fp-runtime`. The `outcome_to_action()` function only exists under `#[cfg(feature = "asupersync")]`. FTUI must not call it unconditionally.

---

## Type Inventory

### FTUI Framework Types (implemented foundation + planned interactive layer)

- Implemented: `FtuiDataSource`, `FsFtuiDataSource`, `ConformanceDashboardSnapshot`, `PacketTrendSnapshot`, `ForensicLogSnapshot`, `GovernanceGateSnapshot`, `DashboardView`, `FtuiAppState`
- Planned next layer: `FtuiApp` event loop/terminal lifecycle, render widgets, theme primitives, input-event abstractions, paginator/virtual-scroller widgets

### Consumed Types from fp-runtime

- `GalaxyBrainCard` -- struct: `title: String`, `equation: String`, `substitution: String`, `intuition: String`
- `DecisionRecord` -- struct: `ts_unix_ms: u64`, `mode: RuntimeMode`, `action: DecisionAction`, `issue: CompatibilityIssue`, `prior_compatible: f64`, `metrics: DecisionMetrics`, `evidence: Vec<EvidenceTerm>`
- `DecisionMetrics` -- struct: `posterior_compatible: f64`, `bayes_factor_compatible_over_incompatible: f64`, `expected_loss_allow: f64`, `expected_loss_reject: f64`, `expected_loss_repair: f64`
- `RuntimeMode` -- enum: `Strict`, `Hardened`
- `DecisionAction` -- enum: `Allow`, `Reject`, `Repair`
- `EvidenceLedger` -- struct: `records: Vec<DecisionRecord>` (append-only)
- `ConformalPredictionSet` -- struct: `quantile_threshold: f64`, `current_score: f64`, `bayesian_action_in_set: bool`, `admissible_actions: Vec<DecisionAction>`, `empirical_coverage: f64`
- `ConformalGuard` -- struct: rolling window, alpha, coverage counters
- `RuntimePolicy` -- struct: `mode: RuntimeMode`, `fail_closed_unknown_features: bool`, `hardened_join_row_cap: Option<usize>`
- `RaptorQEnvelope` -- struct: `artifact_id`, `artifact_type`, `source_hash` (all String), `raptorq: RaptorQMetadata`, `scrub: ScrubStatus`, `decode_proofs: Vec<DecodeProof>`

### Consumed Types from fp-conformance

- `PacketParityReport` -- struct: `suite: String`, `packet_id: Option<String>`, fixture counts, pass/fail
- `PacketGateResult` -- struct: `packet_id: String`, `pass: bool`, strict/hardened failure counts, reasons
- `PacketDriftHistoryEntry` -- struct: `ts_unix_ms: u64`, `packet_id: String`, `suite: String`, fixture/pass/fail counts, `gate_pass: bool`, `report_hash: String`
- `DifferentialReport` -- struct: `report: PacketParityReport`, `differential_results: Vec<DifferentialResult>`, `drift_summary: DriftSummary`
- `DriftRecord` -- struct: `category: ComparisonCategory`, `level: DriftLevel`, `location: String`, `message: String`
- `DriftLevel` -- enum: `Critical`, `NonCritical`, `Informational`
- `ComparisonCategory` -- enum: `Value`, `Type`, `Shape`, `Index`, `Nullness`
- `ForensicLog` -- struct: `events: Vec<ForensicEvent>`
- `ForensicEvent` -- struct: `ts_unix_ms: u64`, `event: ForensicEventKind`
- `ForensicEventKind` -- enum: 10 variants (SuiteStart, SuiteEnd, PacketStart, PacketEnd, CaseStart, CaseEnd, ArtifactWritten, GateEvaluated, DriftHistoryAppended, Error)
- `FailureForensicsReport` -- struct: `run_ts_unix_ms: u64`, fixture counts, `failures: Vec<FailureDigest>`, `gate_failures: Vec<String>`
- `FailureDigest` -- struct: `packet_id`, `case_id`, `operation`, `mode`, `mismatch_summary`, `replay_command`, `artifact_path`
- `ArtifactId` -- struct: `packet_id: String`, `artifact_kind: String`, `run_ts_unix_ms: u64`
- `CiPipelineResult` -- struct: `gates: Vec<CiGateResult>`, `all_passed: bool`, `first_failure: Option<CiGate>`, `elapsed_ms: u64`
- `CiGateResult` -- struct: `gate: CiGate`, `passed: bool`, `elapsed_ms: u64`, `summary: String`, `errors: Vec<String>`
- `CiGate` -- enum: G1Compile, G2Lint, G3Unit, G4Property, G4_5Fuzz, G5Integration, G6Conformance, G7Coverage, G8E2e
- `E2eReport` -- struct: `suite`, `packet_reports`, `artifacts_written`, `gate_results`, `gates_pass`, `drift_history_path`, `forensic_log`, fixture counts
- `WrittenPacketArtifacts` -- struct: `packet_id`, 5 PathBuf fields for artifact files
- `DecodeProofArtifact` -- struct: `packet_id`, `decode_proofs`, `status: DecodeProofStatus`
- `DecodeProofStatus` -- enum: `Recovered`, `Failed`, `NotAttempted`

---

## Rule Ledger

1. **GalaxyBrainCard rendering contract (fp-runtime, lines 97-121):**
   - 1a. `decision_to_card(record)` always produces a card with non-empty `title`, `equation`, `substitution`, and `intuition`.
   - 1b. `title` format is `{issue.subject}::{action:?}` -- always contains `::`.
   - 1c. `equation` is the fixed string `"argmin_a ... P(s|evidence)"` -- FTUI can rely on this for layout sizing.
   - 1d. `substitution` contains `P(compatible|e)=`, `E[allow]=`, `E[reject]=`, `E[repair]=` with 4-decimal precision.
   - 1e. `intuition` is the fixed string about expected loss -- FTUI can hard-wrap at known length.
   - 1f. `render_plain()` is the canonical text format: `[{title}]\n{equation}\n{substitution}\n{intuition}` -- 4 lines, no trailing newline.
   - 1g. `GalaxyBrainCard` implements `Serialize`/`Deserialize` -- FTUI may persist cards as JSON.

2. **Dashboard view taxonomy (planned, derived from TODO N3-N5):**
   - 2a. **Conformance dashboard:** per-packet cards showing gate state, fixture pass/fail ratio, drift trend sparkline.
   - 2b. **Performance dashboard:** round-by-round optimization evidence (Round 2-5 artifacts), p50/p95/p99 latency display, memory budget tracking.
   - 2c. **Forensics dashboard:** failure digest list with replay commands, forensic event timeline, artifact cross-reference index.
   - 2d. **Decision dashboard:** evidence ledger entries, galaxy-brain cards, conformal guard status, Bayesian posterior visualization.
   - 2e. Each dashboard view must be independently scrollable and navigable via keyboard.

3. **Artifact file access pattern:**
   - 3a. FTUI reads artifact JSON files lazily -- only when a dashboard panel requires the data.
   - 3b. FTUI must not hold file locks on artifact files (conformance runs may write concurrently).
   - 3c. FTUI must re-read artifact files on refresh (manual or periodic) to capture updates from in-progress runs.
   - 3d. Artifact paths follow the deterministic scheme: `artifacts/phase2c/{packet_id}/{artifact_file}` per the mapping in FAILURE_FORENSICS_UX.md section 4.2.
   - 3e. `drift_history.jsonl` is read line-by-line, parsed per-line, with malformed lines skipped and counted.

4. **CI gate pipeline rendering:**
   - 4a. Gates are rendered in `CiGate::order()` sequence (G1=1 through G8=9).
   - 4b. Each gate shows its `label()` string, pass/fail status, and `elapsed_ms`.
   - 4c. On failure: gate `summary` and `errors` list are expandable in drilldown.
   - 4d. `CiPipelineResult::first_failure` highlights the blocking gate.
   - 4e. Pipeline total elapsed time and passed/total counts shown in header.

5. **Conformal guard rendering:**
   - 5a. When `is_calibrated()` is false, display "Uncalibrated (N/2 scores)" with `calibration_count()`.
   - 5b. When calibrated, display `conformal_quantile()` value, `empirical_coverage()`, and `coverage_alert()` flag.
   - 5c. If `coverage_alert()` is true (total >= 100 and coverage < 1 - alpha), render a warning indicator.
   - 5d. `alpha` is clamped to `[0.01, 0.5]` -- FTUI can assume valid alpha for display.

6. **Forensic event display rules:**
   - 6a. Events are ordered by `ts_unix_ms` (chronological).
   - 6b. `ForensicEventKind::Error` events must be highlighted (distinct color or icon).
   - 6c. `ForensicEventKind::GateEvaluated` with `pass: false` must be highlighted.
   - 6d. `ForensicEventKind::ArtifactWritten` should be navigable (clicking/selecting opens the artifact path).
   - 6e. Events must be filterable by kind (checkbox or toggle per variant).

7. **Drift history trend rendering:**
   - 7a. X-axis is `ts_unix_ms` (time series), Y-axis is pass rate (`passed / fixture_count`).
   - 7b. Each `PacketDriftHistoryEntry` is a data point.
   - 7c. `gate_pass: false` entries must be visually distinguished (red dot or marker).
   - 7d. Multiple packet IDs must be distinguishable (separate lines or color-coded).
   - 7e. `report_hash` enables deduplication -- identical report hashes indicate reruns of same fixtures with same results.

8. **Failure digest display rules (per FAILURE_FORENSICS_UX.md):**
   - 8a. Format: `FAIL {packet_id}::{case_id} [{operation:?}/{mode:?}]` as header line.
   - 8b. `mismatch_summary` truncated to first 200 characters with ellipsis.
   - 8c. `replay_command` shown verbatim and copy-to-clipboard capable.
   - 8d. `artifact_path` shown only when `Some(_)`, rendered as a navigable link.
   - 8e. Gate failures listed separately after individual case failures.

9. **Strict/hardened mode provenance display (planned, N5):**
   - 9a. Display active `RuntimeMode` (Strict or Hardened) prominently.
   - 9b. Show `fail_closed_unknown_features` boolean with its effect ("rejects all unknown features" vs "allows with evidence").
   - 9c. Show `hardened_join_row_cap` value (or "unlimited" when `None`).
   - 9d. Show evidence ledger record count per mode.
   - 9e. Mode toggle must update the display in real time if FTUI supports live policy changes.

10. **Asupersync + FTUI contract compatibility (planned, N6):**
    - 10a. FTUI must function identically whether `asupersync` feature is enabled or disabled in `fp-runtime`.
    - 10b. When `asupersync` is enabled, FTUI may display additional `outcome_to_action()` mappings in the decision dashboard.
    - 10c. When `asupersync` is disabled, FTUI must suppress asupersync-specific panels without error.
    - 10d. Integration tests must validate FTUI renders correctly under both feature configurations.

---

## Error Ledger

1. **Artifact file not found:**
   - Trigger: FTUI attempts to read an artifact JSON file that does not exist at the expected path.
   - Handling: Display "artifact not found: {path}" in the relevant panel. Do not crash. Do not propagate error to other panels.
   - Impact: The specific packet's detail view is degraded but the dashboard remains functional.

2. **Malformed artifact JSON:**
   - Trigger: An artifact file contains invalid JSON or does not conform to its schema.
   - Handling: Display "parse error: {path}: {error}" in the relevant panel. Skip the artifact and continue rendering other data.
   - Impact: One packet's data may be missing or partial; other packets are unaffected.

3. **Malformed drift history JSONL:**
   - Trigger: A line in `drift_history.jsonl` is not valid JSON or does not deserialize to `PacketDriftHistoryEntry`.
   - Handling: Skip the malformed line. Increment a "skipped lines" counter displayed in the trend view footer.
   - Impact: Drift trend may have gaps but remains displayable.

4. **Terminal too small for dashboard:**
   - Trigger: Terminal dimensions are insufficient to render the requested dashboard layout.
   - Handling: Display a "terminal too small (min: WxH)" message and refuse to render until resized.
   - Impact: Dashboard is temporarily unavailable. FTUI resumes automatically on resize.

5. **Clock skew in decision records (ts_unix_ms = 0):**
   - Trigger: `DecisionRecord::ts_unix_ms` is `0` due to `now_unix_ms()` clock skew fallback.
   - Handling: Display timestamp as "unknown" or "epoch" instead of "1970-01-01". Do not sort these records as earliest.
   - Impact: Chronological ordering may be degraded for affected records.

6. **Empty evidence ledger:**
   - Trigger: `EvidenceLedger::records()` returns an empty slice.
   - Handling: Display "No decisions recorded" in the decision dashboard. Suppress galaxy-brain card panel.
   - Impact: Decision dashboard shows minimal content but does not error.

7. **Conformal guard coverage alert:**
   - Trigger: `ConformalGuard::coverage_alert()` returns true (empirical coverage has dropped below `1 - alpha` after 100+ decisions).
   - Handling: Display a persistent warning badge in the conformal guard panel with the current coverage percentage.
   - Impact: Operator attention is drawn to calibration drift.

8. **IO error reading artifact directory:**
   - Trigger: `artifacts/phase2c/` directory does not exist or is not readable.
   - Handling: Display "artifact directory not accessible: {path}" at the dashboard level. All packet panels show "no data".
   - Impact: Entire conformance dashboard is degraded. Other dashboards (decision, performance) may still function from in-memory data.

9. **Feature flag mismatch (asupersync panel):**
   - Trigger: FTUI code references `outcome_to_action()` but `asupersync` feature is not enabled.
   - Handling: Compile-time error if not gated with `#[cfg(feature = "asupersync")]`. FTUI code must use conditional compilation for asupersync-dependent panels.
   - Impact: Build failure if improperly gated. Correct gating results in graceful panel absence.

10. **Oversized forensic log rendering:**
    - Trigger: `ForensicLog` contains > 10,000 events, causing render lag.
    - Handling: FTUI must implement virtual scrolling or pagination. Display visible window only. Footer shows "N of M events".
    - Impact: Large logs remain navigable without UI freeze.

---

## Hidden Assumptions

1. **`fp-frankentui` now exists but remains a foundation crate, not a full interactive TUI.** FrankenPandas now includes read-only FTUI data loaders and dashboard snapshots in `crates/fp-frankentui`, but terminal backend integration (ratatui/crossterm/termion/termwiz), interactive rendering, and event loop behavior are still open.

2. **Implemented FTUI surfaces are data/snapshot primitives plus plain rendering helpers, not widget-level UI.** `GalaxyBrainCard::render_plain()` and `ConformanceDashboardSnapshot::render_plain()` are text-level interfaces; no full panel layout/styling engine is implemented yet.

3. **All conformance types implement Serialize/Deserialize.** FTUI assumes it can read all artifact files as JSON and deserialize them into the corresponding Rust types. This is true for all types listed in the Type Inventory (verified via `#[derive(Serialize, Deserialize)]` annotations throughout fp-conformance and fp-runtime).

4. **Artifact files use UTF-8 encoding.** All JSON artifacts are produced by `serde_json::to_string()` which outputs UTF-8. FTUI can assume UTF-8 without BOM.

5. **The CiPipelineResult Display impl defines the canonical text format.** FTUI's pipeline rendering should be a graphical enhancement of the existing text format (`CI PIPELINE: ALL GREEN (N/N gates passed in Nms)` with per-gate `[PASS]`/`[FAIL]` lines).

6. **FailureForensicsReport Display impl defines the canonical failure output.** FTUI's forensics view should be a navigable, clickable enhancement of the existing text format (`FAILURES: N/M fixtures failed` with numbered failure digests).

7. **The drift history file is append-only and never truncated.** FTUI can perform incremental reads by seeking to the last-read position. However, no file-position tracking mechanism currently exists.

8. **No async runtime in FTUI.** Following `fp-runtime`'s synchronous design, FTUI is expected to use synchronous file IO and a synchronous event loop (consistent with `frankentui`'s CLI shell role in FrankenSQLite). If frankentui provides async support via asupersync, this assumption must be revisited.

9. **FTUI does not modify artifact files.** It is a read-only consumer. Write operations belong to `fp-conformance` (artifact writers) and `fp-runtime` (evidence ledger). FTUI never writes to `drift_history.jsonl`, parity reports, or any other artifact.

10. **Packet IDs follow the `FP-P2C-NNN` naming convention.** FTUI can use this pattern for sorting, grouping, and display. Currently implemented: FP-P2C-001 through FP-P2C-011.

11. **The 9-gate CI pipeline is the complete gate set.** `CiGate::pipeline()` returns 8 gates (excludes G4_5Fuzz). `CiGate::commit_pipeline()` returns 6 gates (excludes G4_5Fuzz, G7Coverage, G8E2e). FTUI must respect the pipeline variant in use.

12. **No authentication or authorization for FTUI.** The operator cockpit is assumed to run locally with full access to the artifact directory. No multi-user, role-based, or remote access model is specified.

---

## Undefined-Behavior Edges

1. **Frankentui crate API surface:** The frankentui crate has no published API, no documentation, and no source code visible to FrankenPandas. The widget model, layout system, event handling, styling primitives, and terminal backend are entirely unspecified. Whether it uses crossterm, termion, termwiz, or a custom backend is unknown.

2. **FTUI crate structure in FrankenPandas workspace:** Whether FTUI will be a new crate (`fp-tui` or `fp-cockpit`) in the FrankenPandas workspace, a binary within an existing crate, or an external tool is unspecified. The dependency edges to `fp-conformance` and `fp-runtime` are clear but the crate topology is open.

3. **Live vs. snapshot data model:** Whether FTUI operates on static artifact file snapshots (read once, display) or provides live-updating views (watching for file changes, re-reading on modification) is unspecified. The TODO items suggest dashboard-style live monitoring but no file-watching mechanism is defined.

4. **Performance dashboard data sources:** The performance dashboard is mentioned conceptually but no performance artifact types are defined beyond the Round 2-5 markdown files. There are no structured JSON artifacts for latency baselines, memory budgets, or throughput measurements that FTUI could consume programmatically.

5. **Keyboard navigation scheme:** The key bindings for dashboard switching, panel focus, scrolling, drilldown entry/exit, and help display are unspecified. Whether FTUI uses vim-style, emacs-style, or application-specific keybindings is open.

6. **Color and accessibility requirements:** Whether FTUI must support color-blind-safe palettes, high-contrast modes, or monochrome terminals is unspecified. The FrankenSQLite spec mentions syntax highlighting for `fsqlite-cli` but provides no color scheme specification.

7. **Refresh and polling interval:** For live-updating views, the polling interval for artifact file changes and drift history updates is unspecified. Whether FTUI uses inotify/kqueue file watchers, periodic polling, or manual refresh only is open.

8. **Multi-packet comparison views:** Whether FTUI supports side-by-side comparison of two packets (e.g., FP-P2C-001 vs. FP-P2C-002), or cross-packet correlation analysis (e.g., which packets fail together) is unspecified.

9. **Export and clipboard support:** Whether FTUI supports exporting dashboard views to files (HTML, SVG, PNG, text), copying cell values or failure digests to clipboard, or piping output to other tools is unspecified.

10. **Configuration persistence:** Whether FTUI saves user preferences (active dashboard, panel sizes, filter state, sort order) across sessions is unspecified. No configuration file format or location is defined.

11. **Integration testing strategy for FTUI + asupersync (N6):** The TODO item N6 calls for "integration tests that validate asupersync + FTUI contract compatibility under packet drift" but the test methodology, fixture design, and pass/fail criteria are entirely unspecified.

12. **Error escalation path:** When FTUI encounters errors (missing files, parse failures, terminal issues), whether it logs to stderr, writes to a log file, emits forensic events to its own ForensicLog, or silently degrades is unspecified.

13. **Startup and shutdown lifecycle:** How FTUI initializes (loading artifact directory, parsing drift history, calibrating conformal guard), what it displays during loading, and how it handles SIGINT/SIGTERM for clean shutdown is unspecified.

14. **Remote artifact access:** Whether FTUI ever needs to read artifacts from a remote location (S3, HTTP, git remote) is unspecified. The current design assumes local filesystem access but the asupersync integration may introduce remote artifact sync requirements.

---

## Changelog

- **bd-2gi.28.1** (2026-02-14): Initial FRANKENTUI anchor map. Documents the planned TUI operator cockpit for FrankenPandas conformance, performance, and forensics dashboards. Covers legacy anchors (frankentui crate, FrankenSQLite reference, fp-runtime GalaxyBrainCard), behavioral contract (22 conditions across normal/edge/adversarial), type inventory (7 planned FTUI types, 10 consumed fp-runtime types, 24 consumed fp-conformance types), rule ledger (10 numbered rules with sub-items), error ledger (10 error scenarios), hidden assumptions (12 items), and undefined-behavior edges (14 open questions).
- **bd-2gi.28** (2026-02-15): Anchor-map refresh after landing `fp-frankentui` foundation crate. Updated integration points, behavioral contract status, type inventory, and hidden assumptions to reflect implemented data-source/dashboard/app-state scaffolding while keeping interactive rendering scope explicitly planned.
