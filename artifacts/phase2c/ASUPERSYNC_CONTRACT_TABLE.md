# ASUPERSYNC Contract Table + Strict/Hardened Policy Matrix

Bead: `bd-2gi.27.2` [ASUPERSYNC-B]
Subsystem: `fp-runtime` asupersync outcome bridge + artifact synchronization integration
Source anchor: `ASUPERSYNC_ANCHOR_MAP.md` (bd-2gi.27.1)

---

## 1. Summary Contract (Tabular)

| Field | Contract |
|---|---|
| packet_id | `ASUPERSYNC-B` |
| input_contract | `asupersync::Outcome<T, E>` 4-valued lattice (feature-gated); `CompatibilityIssue` + evidence terms + priors for decision engine |
| output_contract | `DecisionAction` enum (`Allow` / `Reject` / `Repair`); `DecisionRecord` with full provenance; `ConformalPredictionSet` for calibrated guard |
| error_contract | `RuntimeError::ClockSkew` on pre-UNIX-epoch clocks; compile-time error when feature flag absent; planned `Cx` cancellation / budget exhaustion errors |
| null_contract | Clock fallback produces `ts_unix_ms = 0` (silent sentinel); no null/Option values in outcome mapping; placeholder envelopes carry zero-valued metadata |
| strict_mode_policy | fail-closed on unknown features; Bayesian argmin overridden to `Reject` when `fail_closed_unknown_features = true` |
| hardened_mode_policy | bounded repair permitted; join admission capped by `hardened_join_row_cap`; all decisions logged to evidence ledger |
| excluded_scope | full Cx capability threading; async/await runtime integration; actual RaptorQ encode/decode pipeline; lab runtime deterministic scheduling; FTUI surfaces |
| oracle_tests | `fp-runtime` unit tests: strict fail-closed, hardened repair, placeholder envelope, conformal calibration suite (AG-09-T) |
| performance_sentinels | Bayesian `decide()` is O(n) in evidence terms; conformal guard is O(w log w) in window size; outcome bridge is O(1) constant-time mapping |
| compatibility_risks | `Outcome::Err` unconditionally maps to `Repair` (no severity distinction); panic payload discarded; clock skew silently produces zero timestamps; no authenticated symbols |
| raptorq_artifacts | `RaptorQEnvelope` placeholder metadata only; actual encode/decode deferred to full asupersync pipeline wiring |

---

## 2. Input Contract (Detailed)

### 2.1 Outcome Bridge (`outcome_to_action`)

| Parameter | Type | Constraint | Notes |
|---|---|---|---|
| `outcome` | `&asupersync::Outcome<T, E>` | Must be one of 4 variants | By-reference; does not consume |
| Feature gate | `#[cfg(feature = "asupersync")]` | Must be explicitly enabled | `default = []` in Cargo.toml |
| `T` | Generic | Any type | Payload ignored by bridge |
| `E` | Generic | Any type | Error payload ignored by bridge |

**Preconditions:**
- The `asupersync` feature flag must be enabled at compile time.
- Caller must gate their own code with `#[cfg(feature = "asupersync")]`.
- `Outcome` must be one of exactly four variants: `Ok`, `Err`, `Cancelled`, `Panicked`.

### 2.2 Decision Engine (`decide()` / `RuntimePolicy` methods)

| Parameter | Type | Constraint | Notes |
|---|---|---|---|
| `mode` | `RuntimeMode` | `Strict` or `Hardened` | Determines fail-closed behavior |
| `issue` | `CompatibilityIssue` | Non-empty `subject` and `detail` | `kind` from `IssueKind` enum |
| `prior_compatible` | `f64` | Must be in (0.0, 1.0) exclusive | Log-odds undefined at 0 or 1 |
| `loss` | `LossMatrix` | All 6 fields must be finite `f64` | Default provided via `Default` impl |
| `evidence` | `Vec<EvidenceTerm>` | Each term has two `f64` log-likelihoods | Empty vec is valid (prior dominates) |
| `estimated_rows` | `usize` | Non-negative | For join admission only |
| `ledger` | `&mut EvidenceLedger` | Mutable reference | Record appended after decision |

**Preconditions:**
- `prior_compatible` must not be 0.0 or 1.0 (log-odds computation would produce infinity/NaN).
- Log-likelihood values should be finite (no NaN, no infinity).
- Evidence term `name` should be non-empty for audit trail clarity.

### 2.3 RaptorQ Envelope (`RaptorQEnvelope::placeholder()`)

| Parameter | Type | Constraint | Notes |
|---|---|---|---|
| `artifact_id` | `impl Into<String>` | Non-empty recommended | Identifies the artifact |
| `artifact_type` | `impl Into<String>` | Non-empty recommended | Categorizes the artifact |

### 2.4 Conformal Guard (`ConformalGuard::new()`)

| Parameter | Type | Constraint | Notes |
|---|---|---|---|
| `window_size` | `usize` | Must be >= 2 for calibration | Controls rolling window capacity |
| `alpha` | `f64` | Clamped to [0.01, 0.5] | Significance level; 0.1 = 90% coverage |

---

## 3. Output Contract (Detailed)

### 3.1 Outcome Bridge Output

| Output | Type | Values | Guarantee |
|---|---|---|---|
| return value | `DecisionAction` | `Allow`, `Reject`, `Repair` | Exhaustive match; `#[must_use]` |

**Mapping table (exhaustive):**

| `Outcome` Variant | `DecisionAction` | Rationale |
|---|---|---|
| `Ok(T)` | `Allow` | Success; permit operation |
| `Err(E)` | `Repair` | Error assumed recoverable |
| `Cancelled(CancelReason)` | `Reject` | External directive; halt |
| `Panicked(PanicPayload)` | `Reject` | Programming error; halt |

**Invariants:**
- INV-OUTCOME-EXHAUSTIVE: All four `Outcome` variants are mapped. Adding a fifth variant to asupersync triggers a compile error (correct behavior).
- INV-OUTCOME-STATELESS: No side effects, no ledger writes, no state mutation.
- INV-OUTCOME-MUST-USE: Caller must consume the returned `DecisionAction`.

### 3.2 Decision Engine Output

| Output | Type | Content | Guarantee |
|---|---|---|---|
| `DecisionRecord` | struct | Full provenance: timestamp, mode, action, issue, prior, metrics, evidence | Appended to `EvidenceLedger` |
| `DecisionAction` | enum | Bayesian argmin (possibly overridden by policy) | Returned to caller |
| `DecisionMetrics` | struct | `posterior_compatible`, `bayes_factor`, expected losses for all 3 actions | Embedded in record |

**Post-conditions:**
- Exactly one `DecisionRecord` is pushed to the ledger per decision call.
- `ts_unix_ms` is the system clock at decision time (or 0 on clock skew).
- `action` reflects Bayesian argmin unless overridden by strict/hardened policy.

### 3.3 Conformal Guard Output

| Output | Type | Content | Guarantee |
|---|---|---|---|
| `ConformalPredictionSet` | struct | threshold, score, in-set flag, admissible actions, empirical coverage | Returned per evaluation |
| `empirical_coverage()` | `f64` | Running coverage rate | 1.0 when no decisions evaluated |
| `coverage_alert()` | `bool` | `true` when coverage < target and >= 100 decisions | No alert under 100 decisions |

### 3.4 RaptorQ Envelope Output

| Output | Type | Content | Guarantee |
|---|---|---|---|
| `RaptorQEnvelope` | struct | Placeholder metadata | `source_hash = "blake3:placeholder"`, `k = 0`, empty proofs |

---

## 4. Error Contract (Detailed)

### 4.1 Enumerated Errors (Current)

| Error | Trigger | Handling | Caller Impact |
|---|---|---|---|
| `RuntimeError::ClockSkew` | `SystemTime::now()` returns pre-UNIX-epoch | `now_unix_ms()` returns `Err(ClockSkew)` | `decide()` catches via `unwrap_or_default()`, producing `ts_unix_ms = 0` |
| Compile error (missing feature) | Calling `outcome_to_action()` without `asupersync` feature | Function does not exist; compile fails | Caller must gate with `#[cfg(feature = "asupersync")]` |

### 4.2 Planned Errors (Asupersync Integration)

| Error | Trigger | Planned Handling | FrankenPandas Impact |
|---|---|---|---|
| `Cx` cancellation | Parent region cancelled during FP operation | `Cx::checkpoint()` returns `ErrorKind::Cancelled` | Operation must abort cleanly; no partial evidence committed |
| Budget exhaustion (deadline) | Operation exceeds wall-clock deadline | `cx.scope_with_budget()` enforcement | Long-running alignment/groupby/join must respect budget |
| Budget exhaustion (poll quota) | Operation exceeds poll count | `cx.scope_with_budget()` enforcement | Kernel iteration must checkpoint periodically |
| Budget exhaustion (cost quota) | Operation exceeds cost units | `cx.scope_with_budget()` enforcement | Memory-intensive operations must track cost |
| RaptorQ decode failure | Insufficient repair symbols | `DecodeProof` with failure reason emitted | Artifact bundles become unrecoverable |
| Obligation leak | Reserved obligation dropped without commit/abort | Lab: panic; Production: diagnostic + escalate | Evidence ledger push may need two-phase obligation |

### 4.3 Error Propagation Rules

| Source | Error Kind | Propagation Strategy |
|---|---|---|
| `outcome_to_action` | None (infallible mapping) | N/A |
| `now_unix_ms()` | `RuntimeError::ClockSkew` | Silently defaults to 0 in `decide()` |
| `LossMatrix` arithmetic | NaN/Inf from degenerate inputs | No guard; caller responsible for finite inputs |
| `ConformalGuard::conformal_quantile()` | Insufficient data | Returns `None`; guard admits all actions |
| Feature flag | Compile-time | No runtime fallback |

---

## 5. Null/Missing Contract (Detailed)

| Scenario | Null Representation | Behavior | Risk |
|---|---|---|---|
| Clock skew | `ts_unix_ms = 0` | Silent sentinel; no crash, no log | Decision records with timestamp 0 are indistinguishable from genuine epoch-0 records |
| Empty evidence vector | `Vec::new()` | Prior dominates posterior; LLR sum = 0.0 | Valid but decision based solely on prior + loss matrix |
| Placeholder RaptorQ envelope | `k = 0`, `repair_symbols = 0`, empty hashes | Structurally valid; no decode capability | Cannot reconstruct any artifact from placeholder |
| Placeholder scrub status | `last_ok_unix_ms = 0`, `status = "placeholder"` | Explicitly marks placeholder envelope as never scrubbed | Consumers must treat as sentinel metadata, not a verified scrub result |
| Uncalibrated conformal guard | `conformal_quantile() = None` | All actions admitted; threshold = infinity | No conformal protection until >= 2 scores |
| No hardened join row cap | `hardened_join_row_cap = None` | `usize::MAX` used; effectively no cap | All joins admitted regardless of cardinality estimate |
| Panic payload in `Outcome::Panicked` | Payload exists but is discarded | Maps to `Reject` without inspection | Diagnostic information from panic is lost at policy layer |

---

## 6. Strict Mode Policy Matrix

| Domain | Strict Mode Behavior | Mechanism | Override Source |
|---|---|---|---|
| Unknown features | **REJECT** (fail-closed) | `fail_closed_unknown_features = true` overrides Bayesian argmin | `RuntimePolicy::strict()` constructor |
| Malformed input | **Bayesian argmin** | No special override | `decide()` output |
| Join cardinality | **Bayesian argmin** | No cap in strict mode (`hardened_join_row_cap = None`) | `decide()` output |
| Policy override | **Bayesian argmin** | No special override | `decide()` output |
| Outcome::Ok | **ALLOW** | Direct mapping | `outcome_to_action()` |
| Outcome::Err | **REPAIR** | Direct mapping | `outcome_to_action()` |
| Outcome::Cancelled | **REJECT** | Direct mapping | `outcome_to_action()` |
| Outcome::Panicked | **REJECT** | Direct mapping | `outcome_to_action()` |
| Clock skew | **SILENT ZERO** | `ts_unix_ms = 0` | `unwrap_or_default()` |
| Conformal guard uncalibrated | **ADMIT ALL** | Threshold = infinity | `evaluate()` fallback |
| Conformal guard calibrated | **Bayesian if in set; widen if not** | Score vs quantile threshold | `evaluate()` logic |
| Evidence ledger | **APPEND-ONLY** | `push()` always succeeds | No filtering or rejection of records |

**Strict mode invariants:**
- SINV-FAIL-CLOSED: Unknown features are always rejected regardless of Bayesian posterior.
- SINV-AUDIT: Every decision produces a `DecisionRecord` in the evidence ledger.
- SINV-NO-REPAIR-UNKNOWN: `decide_unknown_feature()` in strict mode never returns `Allow` or `Repair`.
- SINV-DETERMINISTIC: Given the same inputs, `decide()` produces the same `DecisionAction` (modulo clock).

---

## 7. Hardened Mode Policy Matrix

| Domain | Hardened Mode Behavior | Mechanism | Override Source |
|---|---|---|---|
| Unknown features | **Bayesian argmin** (repair permitted) | `fail_closed_unknown_features = false` | `RuntimePolicy::hardened()` constructor |
| Malformed input | **Bayesian argmin** | No special override | `decide()` output |
| Join cardinality (under cap) | **Bayesian argmin** | Estimated rows <= cap | `decide()` output |
| Join cardinality (over cap) | **REPAIR** (forced) | `estimated_rows > cap` overrides Bayesian argmin to `Repair` | `decide_join_admission()` post-decision override |
| Policy override | **Bayesian argmin** | No special override | `decide()` output |
| Outcome::Ok | **ALLOW** | Direct mapping | `outcome_to_action()` |
| Outcome::Err | **REPAIR** | Direct mapping | `outcome_to_action()` |
| Outcome::Cancelled | **REJECT** | Direct mapping | `outcome_to_action()` |
| Outcome::Panicked | **REJECT** | Direct mapping | `outcome_to_action()` |
| Clock skew | **SILENT ZERO** | `ts_unix_ms = 0` | `unwrap_or_default()` |
| Conformal guard uncalibrated | **ADMIT ALL** | Threshold = infinity | `evaluate()` fallback |
| Conformal guard calibrated | **Bayesian if in set; widen if not** | Score vs quantile threshold | `evaluate()` logic |
| Evidence ledger | **APPEND-ONLY** | `push()` always succeeds | No filtering or rejection of records |

**Hardened mode invariants:**
- HINV-BOUNDED-REPAIR: Join admission is forced to `Repair` when estimated rows exceed `hardened_join_row_cap`.
- HINV-AUDIT: Every decision produces a `DecisionRecord` in the evidence ledger.
- HINV-REPAIR-PREFERRED: The loss matrix for join admission has `repair_if_compatible = 1.5` (lower than `reject_if_compatible = 5.0`), biasing toward repair.
- HINV-CAP-FALLBACK: When `hardened_join_row_cap = None`, no cap is enforced (`usize::MAX` used).

---

## 8. Strict vs Hardened Comparison Matrix

| Property | Strict | Hardened | Delta |
|---|---|---|---|
| `fail_closed_unknown_features` | `true` | `false` | Strict always rejects unknown; hardened uses Bayesian |
| `hardened_join_row_cap` | `None` | `Some(cap)` or `None` | Hardened can bound join cardinality |
| Unknown feature decision | Always `Reject` | Bayesian argmin (may be `Allow`, `Repair`, or `Reject`) | Hardened is more permissive |
| Over-cap join decision | N/A (no cap) | Forced `Repair` | Hardened caps large joins |
| Under-cap join decision | Bayesian argmin | Bayesian argmin | Same |
| Default constructor | `RuntimePolicy::strict()` | `RuntimePolicy::hardened(join_row_cap)` | Different defaults |
| `Default` trait impl | `strict()` | N/A | Strict is the default policy |
| Outcome bridge | Same | Same | Mode-independent |
| Conformal guard | Same | Same | Mode-independent |
| Evidence ledger | Same | Same | Mode-independent |

---

## 9. Performance Sentinels

### 9.1 Complexity Bounds

| Operation | Time Complexity | Space Complexity | Notes |
|---|---|---|---|
| `outcome_to_action()` | O(1) | O(1) | Pattern match on 4 variants; zero allocation |
| `decide()` | O(n) where n = evidence terms | O(n) for evidence vec | LLR sum is linear scan; no sorting |
| `decide_unknown_feature()` | O(1) | O(1) | Fixed 2 evidence terms |
| `decide_join_admission()` | O(1) | O(1) | Fixed 2 evidence terms |
| `ConformalGuard::evaluate()` | O(1) amortized | O(w) where w = window_size | Score insert is O(1); quantile computed lazily |
| `ConformalGuard::conformal_quantile()` | O(w log w) | O(w) | Clone + sort of window |
| `EvidenceLedger::push()` | O(1) amortized | O(1) amortized | Vec append |
| `RaptorQEnvelope::placeholder()` | O(1) | O(1) | Fixed-size struct construction |
| `decision_to_card()` | O(1) | O(1) | String formatting |
| `nonconformity_score()` | O(1) | O(1) | Clamp + ln + abs |

### 9.2 Budget Ceilings (Current)

| Resource | Budget | Enforcement | Status |
|---|---|---|---|
| Wall-clock deadline | None | No enforcement | Gap: operations run unbounded |
| Poll quota | None | No enforcement | Gap: no cooperative checkpoint calls |
| Cost quota | None | No enforcement | Gap: no cost accounting |
| Memory (evidence ledger) | Unbounded | `Vec` grows without limit | Risk: long-running sessions accumulate records |
| Memory (conformal window) | Bounded by `window_size` | Rolling eviction at capacity | OK: controlled by constructor parameter |
| Allocations per decision | 2-3 `String` allocs + Vec | No pooling or arena | Acceptable for policy-layer frequency |

### 9.3 Planned Budget Integration (Asupersync Cx)

| Resource | Planned Mechanism | Integration Point |
|---|---|---|
| Wall-clock deadline | `cx.scope_with_budget(Budget { deadline: ..., .. })` | Wrap FP operation entry points |
| Poll quota | `cx.checkpoint()` at kernel iteration boundaries | Alignment loop, groupby aggregation, join expansion |
| Cost quota | `cx.scope_with_budget(Budget { cost_quota: ..., .. })` | Memory-intensive operations |
| Priority | `Budget.priority` (join semantics) | Task scheduling in async context |

---

## 10. Compatibility Risks

### 10.1 Current Risks

| Risk ID | Description | Severity | Mitigation |
|---|---|---|---|
| CR-01 | `Outcome::Err` universally maps to `Repair` regardless of error severity | Medium | Future: inspect error payload or use error classification |
| CR-02 | `Outcome::Panicked` payload discarded at bridge; diagnostic info lost | Low | Future: extract panic message into `DecisionRecord.issue.detail` |
| CR-03 | `Outcome::Cancelled` and `Outcome::Panicked` conflated to same action | Low | Intentional design; distinction preserved in `Outcome` type upstream |
| CR-04 | Clock skew produces silent `ts_unix_ms = 0` without logging | Medium | Future: log warning on clock skew; route through `Cx` time capability |
| CR-05 | `prior_compatible` at 0.0 or 1.0 produces infinity/NaN in log-odds | High | Future: clamp prior to (epsilon, 1-epsilon) in `decide()` |
| CR-06 | Evidence log-likelihoods at NaN/Inf produce garbage posteriors | Medium | Future: validate evidence terms before LLR computation |
| CR-07 | Placeholder RaptorQ envelopes misinterpreted as real encoded artifacts | Low | Sentinel hash `"blake3:placeholder"` distinguishes; callers must check |
| CR-08 | No authenticated symbols for artifact integrity | Medium | Future: integrate `SecurityContext` + `AuthenticatedSymbol` from asupersync |
| CR-09 | `SystemTime::now()` violates INV-NO-AMBIENT-AUTHORITY | High (for lab mode) | Future: route through `Cx` time capability |
| CR-10 | Long-running FP operations cannot be cooperatively cancelled | Medium | Future: insert `cx.checkpoint()` at kernel boundaries |
| CR-11 | Feature flag mismatch causes compile error, not runtime degradation | Low | By design; no runtime fallback is intentional |
| CR-12 | `ConformalGuard` rolling window uses `Vec::remove(0)` (O(n) shift) | Low | Acceptable for window sizes <= 1000; future: use `VecDeque` |

### 10.2 Asupersync Version Coupling

| Aspect | Current State | Risk |
|---|---|---|
| Pinned version | `asupersync = "0.1.1"` (semver compatible range) | Minor version bump may add API surface |
| `Outcome` variant count | 4 variants; exhaustive match | New variant = compile error (correct) |
| `default-features = false` | Minimal surface | Assumes `Outcome` is available without features |
| Transitive dependencies | `base64`, `bincode`, `crossbeam-queue`, `getrandom`, `libc`, `nix`, `parking_lot`, `pin-project`, `polling`, `rmp-serde`, `serde` | Large dependency tree for optional feature |

### 10.3 Divergence from FrankenSQLite Reference

| Aspect | FrankenSQLite Spec | FrankenPandas Current | Gap |
|---|---|---|---|
| `Cx` threading | All operations accept `&Cx` | No `Cx` parameter anywhere | Full gap: no capability context |
| Ambient authority | Prohibited (INV-NO-AMBIENT-AUTHORITY) | `SystemTime::now()` called directly | Violation in `now_unix_ms()` |
| Cancellation checkpoints | At every cooperatively cancellable point | None | Full gap: no checkpoint calls |
| Budget enforcement | Deadline + poll + cost + priority | None | Full gap: no budget awareness |
| Supervision | Outcome-based restart/escalate policy | Bridge maps to DecisionAction only | Partial: mapping exists but no supervision loop |
| Obligation tracking | Two-phase reserve/commit/abort | Ledger `push()` is fire-and-forget | Full gap: no obligation discipline |
| Authenticated symbols | Epoch-scoped auth tags on RaptorQ symbols | None | Full gap: placeholder envelopes only |
| Epoch coordination | `EpochId` for distributed barriers | None | Full gap: no epoch awareness |
| Lab runtime | Deterministic scheduling + oracle suite | None | Full gap: no lab mode |
| TLA+ trace export | `TlaExporter` for model checking | None | Full gap: no trace export |

---

## 11. Planned Integration Roadmap (from TODO N-items)

| Item | Description | Contract Impact | Priority |
|---|---|---|---|
| N1 | Extend outcome bridge with packet gate summaries + mismatch corpus pointers | Expands `outcome_to_action` output; new fields in `DecisionRecord` | High |
| N2 | Deterministic sync schema for conformance/perf artifact bundles | Defines RaptorQ encoding policy per artifact type | High |
| N3 | FTUI packet dashboard cards | Read-only consumer of `DecisionRecord` + `RaptorQEnvelope` | Medium |
| N4 | FTUI drilldown for mismatch corpus replay + evidence traces | Read-only consumer of `EvidenceLedger` | Medium |
| N5 | Strict/hardened mode toggle telemetry in FTUI | Read-only consumer of `RuntimePolicy` + `RuntimeMode` | Low |
| N6 | Integration tests for asupersync + FTUI contract compatibility | Test-only; validates contract stability under packet drift | High |

---

## 12. Machine-Checkable Invariant Summary

| Invariant ID | Statement | Checkable By | Status |
|---|---|---|---|
| INV-OUTCOME-EXHAUSTIVE | `outcome_to_action` match covers all 4 `Outcome` variants | Compiler (exhaustive match) | Enforced |
| INV-OUTCOME-STATELESS | `outcome_to_action` has no side effects | Code review; `&` reference input, no `&mut` | Enforced |
| INV-OUTCOME-MUST-USE | Return value of `outcome_to_action` must be consumed | `#[must_use]` attribute | Enforced |
| INV-FEATURE-ISOLATED | Asupersync dependency is never pulled without explicit opt-in | `default = []` in Cargo.toml | Enforced |
| INV-FEATURE-NO-TRANSITIVE | No other workspace crate depends on asupersync | Cargo.toml audit | Enforced |
| INV-LEDGER-APPEND-ONLY | `EvidenceLedger` only supports `push()`; no delete/modify | API surface (no `&mut` access to `records`) | Enforced |
| INV-CONFORMAL-ALPHA-CLAMPED | `alpha` is clamped to [0.01, 0.5] regardless of input | `.clamp(0.01, 0.5)` in constructor | Enforced |
| INV-CONFORMAL-UNCALIBRATED-SAFE | Guard admits all actions when < 2 calibration scores | `conformal_quantile()` returns `None` path | Enforced |
| INV-STRICT-FAIL-CLOSED | Strict mode always rejects unknown features | Post-decision override in `decide_unknown_feature()` | Enforced |
| INV-HARDENED-CAP-REPAIR | Hardened mode forces `Repair` for over-cap joins | Post-decision override in `decide_join_admission()` | Enforced |
| INV-PLACEHOLDER-SENTINEL | Placeholder RaptorQ envelopes have `source_hash = "blake3:placeholder"` | Constructor implementation | Enforced |
| INV-NO-UNSAFE | `fp-runtime` contains `#![forbid(unsafe_code)]` | Compiler | Enforced |
| INV-NO-AMBIENT-AUTHORITY | No direct calls to ambient side-effect APIs | Code review | **VIOLATED** (`SystemTime::now()` in `now_unix_ms()`) |
| INV-SUPERVISION-MONOTONE | `Panicked` must not be restarted; `Cancelled` must stop | Design rule; no supervision loop yet | Planned |
| INV-OBLIGATION-RESOLVE | Reserved obligations must resolve to Committed or Aborted | Design rule; no obligation model yet | Planned |

---

## 13. Loss Matrix Reference (Default Values)

### 13.1 General Loss Matrix (`LossMatrix::default()`)

| | Compatible (true state) | Incompatible (true state) |
|---|---|---|
| **Allow** (action) | 0.0 | 100.0 |
| **Reject** (action) | 6.0 | 0.5 |
| **Repair** (action) | 2.0 | 3.0 |

**Interpretation:** Allowing an incompatible input is catastrophic (100.0). Rejecting a compatible input is costly (6.0). Repair is a moderate-cost hedge in either state.

### 13.2 Join Admission Loss Matrix

| | Compatible (true state) | Incompatible (true state) |
|---|---|---|
| **Allow** (action) | 0.0 | 130.0 |
| **Reject** (action) | 5.0 | 0.5 |
| **Repair** (action) | 1.5 | 3.0 |

**Interpretation:** Join cardinality explosion is more severe (130.0 vs 100.0). Repair cost for compatible joins is lower (1.5 vs 2.0) because bounded-cap repair is well-defined.

### 13.3 Prior Probabilities

| Decision Domain | Prior P(compatible) | Rationale |
|---|---|---|
| Unknown feature | 0.25 | Low prior: unknown features are likely incompatible |
| Join admission | 0.60 | Moderate prior: most joins are within expected bounds |

---

## 14. Type Contract Cross-Reference

| FP-Runtime Type | Serde Format | Serializable | Clone | PartialEq | Default |
|---|---|---|---|---|---|
| `RuntimeMode` | `snake_case` enum | Yes | Yes (Copy) | Yes | No |
| `DecisionAction` | `snake_case` enum | Yes | Yes (Copy) | Yes | No |
| `IssueKind` | `snake_case` enum | Yes | Yes (Copy) | Yes | No |
| `CompatibilityIssue` | struct | Yes | Yes | Yes (Eq) | No |
| `EvidenceTerm` | struct | Yes | Yes | Yes | No |
| `LossMatrix` | struct | Yes | Yes (Copy) | Yes | Yes |
| `DecisionMetrics` | struct | Yes | Yes | Yes | No |
| `DecisionRecord` | struct | Yes | Yes | Yes | No |
| `GalaxyBrainCard` | struct | Yes (Eq) | Yes | Yes (Eq) | No |
| `EvidenceLedger` | struct | Yes | Yes | Yes | Yes |
| `RuntimePolicy` | struct | No (no Serialize) | Yes | Yes (Eq) | Yes (strict) |
| `RuntimeError` | enum | No | No | No | No |
| `RaptorQEnvelope` | struct | Yes | Yes | Yes | No |
| `RaptorQMetadata` | struct | Yes | Yes | Yes | No |
| `ScrubStatus` | struct | Yes | Yes | Yes | No |
| `DecodeProof` | struct | Yes | Yes | Yes | No |
| `ConformalPredictionSet` | struct | Yes | Yes | Yes | No |
| `ConformalGuard` | struct | Yes | Yes | No (f64 scores) | No |
