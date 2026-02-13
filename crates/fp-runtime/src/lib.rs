#![forbid(unsafe_code)]

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionAction {
    Allow,
    Reject,
    Repair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueKind {
    UnknownFeature,
    MalformedInput,
    JoinCardinality,
    PolicyOverride,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    pub kind: IssueKind,
    pub subject: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceTerm {
    pub name: String,
    pub log_likelihood_if_compatible: f64,
    pub log_likelihood_if_incompatible: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LossMatrix {
    pub allow_if_compatible: f64,
    pub allow_if_incompatible: f64,
    pub reject_if_compatible: f64,
    pub reject_if_incompatible: f64,
    pub repair_if_compatible: f64,
    pub repair_if_incompatible: f64,
}

impl Default for LossMatrix {
    fn default() -> Self {
        Self {
            allow_if_compatible: 0.0,
            allow_if_incompatible: 100.0,
            reject_if_compatible: 6.0,
            reject_if_incompatible: 0.5,
            repair_if_compatible: 2.0,
            repair_if_incompatible: 3.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionMetrics {
    pub posterior_compatible: f64,
    pub bayes_factor_compatible_over_incompatible: f64,
    pub expected_loss_allow: f64,
    pub expected_loss_reject: f64,
    pub expected_loss_repair: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub ts_unix_ms: u64,
    pub mode: RuntimeMode,
    pub action: DecisionAction,
    pub issue: CompatibilityIssue,
    pub prior_compatible: f64,
    pub metrics: DecisionMetrics,
    pub evidence: Vec<EvidenceTerm>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GalaxyBrainCard {
    pub title: String,
    pub equation: String,
    pub substitution: String,
    pub intuition: String,
}

impl GalaxyBrainCard {
    #[must_use]
    pub fn render_plain(&self) -> String {
        format!(
            "[{}]\n{}\n{}\n{}",
            self.title, self.equation, self.substitution, self.intuition
        )
    }
}

#[must_use]
pub fn decision_to_card(record: &DecisionRecord) -> GalaxyBrainCard {
    GalaxyBrainCard {
        title: format!("{}::{:?}", record.issue.subject, record.action),
        equation: "argmin_a Σ_s L(a,s) P(s|evidence)".to_owned(),
        substitution: format!(
            "P(compatible|e)={:.4}, E[allow]={:.4}, E[reject]={:.4}, E[repair]={:.4}",
            record.metrics.posterior_compatible,
            record.metrics.expected_loss_allow,
            record.metrics.expected_loss_reject,
            record.metrics.expected_loss_repair
        ),
        intuition: "Lower expected loss wins; strict mode may still force fail-closed.".to_owned(),
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceLedger {
    records: Vec<DecisionRecord>,
}

impl EvidenceLedger {
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    pub fn push(&mut self, record: DecisionRecord) {
        self.records.push(record);
    }

    #[must_use]
    pub fn records(&self) -> &[DecisionRecord] {
        &self.records
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePolicy {
    pub mode: RuntimeMode,
    pub fail_closed_unknown_features: bool,
    pub hardened_join_row_cap: Option<usize>,
}

impl RuntimePolicy {
    #[must_use]
    pub fn strict() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            fail_closed_unknown_features: true,
            hardened_join_row_cap: None,
        }
    }

    #[must_use]
    pub fn hardened(join_row_cap: Option<usize>) -> Self {
        Self {
            mode: RuntimeMode::Hardened,
            fail_closed_unknown_features: false,
            hardened_join_row_cap: join_row_cap,
        }
    }

    pub fn decide_unknown_feature(
        &self,
        subject: impl Into<String>,
        detail: impl Into<String>,
        ledger: &mut EvidenceLedger,
    ) -> DecisionAction {
        let issue = CompatibilityIssue {
            kind: IssueKind::UnknownFeature,
            subject: subject.into(),
            detail: detail.into(),
        };

        let evidence = vec![
            EvidenceTerm {
                name: "compatibility_allowlist_miss".to_owned(),
                log_likelihood_if_compatible: -3.5,
                log_likelihood_if_incompatible: -0.2,
            },
            EvidenceTerm {
                name: "unknown_protocol_field".to_owned(),
                log_likelihood_if_compatible: -2.0,
                log_likelihood_if_incompatible: -0.1,
            },
        ];

        let mut record = decide(self.mode, issue, 0.25, LossMatrix::default(), evidence);
        if self.fail_closed_unknown_features {
            record.action = DecisionAction::Reject;
        }
        let action = record.action;
        ledger.push(record);
        action
    }

    pub fn decide_join_admission(
        &self,
        estimated_rows: usize,
        ledger: &mut EvidenceLedger,
    ) -> DecisionAction {
        let issue = CompatibilityIssue {
            kind: IssueKind::JoinCardinality,
            subject: "join_estimator".to_owned(),
            detail: format!("estimated_rows={estimated_rows}"),
        };

        let cap = self.hardened_join_row_cap.unwrap_or(usize::MAX);
        let evidence = vec![
            EvidenceTerm {
                name: "estimator_overflow_risk".to_owned(),
                log_likelihood_if_compatible: if estimated_rows <= cap { -0.3 } else { -2.8 },
                log_likelihood_if_incompatible: if estimated_rows <= cap { -1.2 } else { -0.1 },
            },
            EvidenceTerm {
                name: "memory_budget_signal".to_owned(),
                log_likelihood_if_compatible: if estimated_rows <= cap { -0.4 } else { -2.2 },
                log_likelihood_if_incompatible: if estimated_rows <= cap { -1.5 } else { -0.2 },
            },
        ];

        let loss = LossMatrix {
            allow_if_compatible: 0.0,
            allow_if_incompatible: 130.0,
            reject_if_compatible: 5.0,
            reject_if_incompatible: 0.5,
            repair_if_compatible: 1.5,
            repair_if_incompatible: 3.0,
        };

        let mut record = decide(self.mode, issue, 0.6, loss, evidence);

        if matches!(self.mode, RuntimeMode::Hardened) && estimated_rows > cap {
            record.action = DecisionAction::Repair;
        }

        let action = record.action;
        ledger.push(record);
        action
    }
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self::strict()
    }
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("system clock is before UNIX_EPOCH")]
    ClockSkew,
}

fn now_unix_ms() -> Result<u64, RuntimeError> {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| RuntimeError::ClockSkew)?
        .as_millis();
    Ok(ms as u64)
}

fn decide(
    mode: RuntimeMode,
    issue: CompatibilityIssue,
    prior_compatible: f64,
    loss: LossMatrix,
    evidence: Vec<EvidenceTerm>,
) -> DecisionRecord {
    let log_odds_prior = (prior_compatible / (1.0 - prior_compatible)).ln();
    let llr_sum: f64 = evidence
        .iter()
        .map(|term| term.log_likelihood_if_compatible - term.log_likelihood_if_incompatible)
        .sum();
    let log_odds_post = log_odds_prior + llr_sum;

    let posterior_compatible = 1.0 / (1.0 + (-log_odds_post).exp());
    let posterior_incompatible = 1.0 - posterior_compatible;

    let expected_loss_allow = loss.allow_if_compatible * posterior_compatible
        + loss.allow_if_incompatible * posterior_incompatible;
    let expected_loss_reject = loss.reject_if_compatible * posterior_compatible
        + loss.reject_if_incompatible * posterior_incompatible;
    let expected_loss_repair = loss.repair_if_compatible * posterior_compatible
        + loss.repair_if_incompatible * posterior_incompatible;

    let mut best_action = DecisionAction::Allow;
    let mut best_loss = expected_loss_allow;

    if expected_loss_repair < best_loss {
        best_action = DecisionAction::Repair;
        best_loss = expected_loss_repair;
    }
    if expected_loss_reject < best_loss {
        best_action = DecisionAction::Reject;
    }

    DecisionRecord {
        ts_unix_ms: now_unix_ms().unwrap_or_default(),
        mode,
        action: best_action,
        issue,
        prior_compatible,
        metrics: DecisionMetrics {
            posterior_compatible,
            bayes_factor_compatible_over_incompatible: llr_sum.exp(),
            expected_loss_allow,
            expected_loss_reject,
            expected_loss_repair,
        },
        evidence,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RaptorQEnvelope {
    pub artifact_id: String,
    pub artifact_type: String,
    pub source_hash: String,
    pub raptorq: RaptorQMetadata,
    pub scrub: ScrubStatus,
    pub decode_proofs: Vec<DecodeProof>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RaptorQMetadata {
    pub k: u32,
    pub repair_symbols: u32,
    pub overhead_ratio: f64,
    pub symbol_hashes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScrubStatus {
    pub last_ok_unix_ms: u64,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeProof {
    pub ts_unix_ms: u64,
    pub reason: String,
    pub recovered_blocks: u32,
    pub proof_hash: String,
}

impl RaptorQEnvelope {
    #[must_use]
    pub fn placeholder(artifact_id: impl Into<String>, artifact_type: impl Into<String>) -> Self {
        Self {
            artifact_id: artifact_id.into(),
            artifact_type: artifact_type.into(),
            source_hash: "blake3:placeholder".to_owned(),
            raptorq: RaptorQMetadata {
                k: 0,
                repair_symbols: 0,
                overhead_ratio: 0.0,
                symbol_hashes: Vec::new(),
            },
            scrub: ScrubStatus {
                last_ok_unix_ms: 0,
                status: "ok".to_owned(),
            },
            decode_proofs: Vec::new(),
        }
    }
}

#[cfg(feature = "asupersync")]
#[must_use]
pub fn outcome_to_action<T, E>(outcome: &asupersync::Outcome<T, E>) -> DecisionAction {
    match outcome {
        asupersync::Outcome::Ok(_) => DecisionAction::Allow,
        asupersync::Outcome::Err(_) => DecisionAction::Repair,
        asupersync::Outcome::Cancelled(_) | asupersync::Outcome::Panicked(_) => {
            DecisionAction::Reject
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DecisionAction, EvidenceLedger, RaptorQEnvelope, RuntimeMode, RuntimePolicy,
        decision_to_card,
    };

    #[test]
    fn strict_mode_fails_closed_for_unknown_features() {
        let mut ledger = EvidenceLedger::new();
        let policy = RuntimePolicy::strict();

        let action = policy.decide_unknown_feature("csv", "field=experimental", &mut ledger);
        assert_eq!(action, DecisionAction::Reject);
        assert_eq!(ledger.records()[0].mode, RuntimeMode::Strict);
    }

    #[test]
    fn hardened_mode_repairs_large_join_estimates() {
        let mut ledger = EvidenceLedger::new();
        let policy = RuntimePolicy::hardened(Some(10_000));

        let action = policy.decide_join_admission(100_000, &mut ledger);
        assert_eq!(action, DecisionAction::Repair);
        assert_eq!(ledger.records().len(), 1);
    }

    #[test]
    fn placeholder_raptorq_envelope_is_well_formed() {
        let envelope = RaptorQEnvelope::placeholder("packet-001", "conformance");
        assert_eq!(envelope.artifact_id, "packet-001");
        assert_eq!(envelope.artifact_type, "conformance");
        assert_eq!(envelope.scrub.status, "ok");
    }

    #[test]
    fn decision_card_is_renderable_for_ftui_consumers() {
        let mut ledger = EvidenceLedger::new();
        let policy = RuntimePolicy::strict();
        policy.decide_unknown_feature("csv", "field=experimental", &mut ledger);

        let card = decision_to_card(&ledger.records()[0]);
        let rendered = card.render_plain();
        assert!(rendered.contains("argmin_a"));
        assert!(rendered.contains("P(compatible|e)"));
    }
}
