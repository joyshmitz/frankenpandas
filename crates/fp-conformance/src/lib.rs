#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use fp_frame::{FrameError, Series};
use fp_groupby::{GroupByOptions, groupby_sum};
use fp_index::{AlignmentPlan, Index, IndexLabel, align_union, validate_alignment_plan};
use fp_join::{JoinType, join_series};
use fp_runtime::{
    DecodeProof, EvidenceLedger, RaptorQEnvelope, RaptorQMetadata, RuntimeMode, RuntimePolicy,
    ScrubStatus,
};
use fp_types::Scalar;
use raptorq::{Decoder, Encoder, EncodingPacket, ObjectTransmissionInformation};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub repo_root: PathBuf,
    pub oracle_root: PathBuf,
    pub fixture_root: PathBuf,
    pub strict_mode: bool,
    pub python_bin: String,
    pub allow_system_pandas_fallback: bool,
}

impl HarnessConfig {
    #[must_use]
    pub fn default_paths() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            oracle_root: repo_root.join("legacy_pandas_code/pandas"),
            fixture_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"),
            strict_mode: true,
            python_bin: "python3".to_owned(),
            allow_system_pandas_fallback: false,
            repo_root,
        }
    }

    #[must_use]
    pub fn packet_fixture_root(&self) -> PathBuf {
        self.fixture_root.join("packets")
    }

    #[must_use]
    pub fn packet_artifact_root(&self, packet_id: &str) -> PathBuf {
        self.repo_root.join("artifacts/phase2c").join(packet_id)
    }

    #[must_use]
    pub fn parity_gate_path(&self, packet_id: &str) -> PathBuf {
        self.packet_artifact_root(packet_id)
            .join("parity_gate.yaml")
    }

    #[must_use]
    pub fn oracle_script_path(&self) -> PathBuf {
        self.repo_root
            .join("crates/fp-conformance/oracle/pandas_oracle.py")
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self::default_paths()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessReport {
    pub suite: &'static str,
    pub oracle_present: bool,
    pub fixture_count: usize,
    pub strict_mode: bool,
}

#[must_use]
pub fn run_smoke(config: &HarnessConfig) -> HarnessReport {
    let fixture_count = fs::read_dir(&config.fixture_root)
        .ok()
        .into_iter()
        .flat_map(|it| it.filter_map(Result::ok))
        .count();

    HarnessReport {
        suite: "smoke",
        oracle_present: config.oracle_root.exists(),
        fixture_count,
        strict_mode: config.strict_mode,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleMode {
    FixtureExpected,
    LiveLegacyPandas,
}

#[derive(Debug, Clone)]
pub struct SuiteOptions {
    pub packet_filter: Option<String>,
    pub oracle_mode: OracleMode,
}

impl Default for SuiteOptions {
    fn default() -> Self {
        Self {
            packet_filter: None,
            oracle_mode: OracleMode::FixtureExpected,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureOperation {
    SeriesAdd,
    SeriesJoin,
    #[serde(rename = "groupby_sum", alias = "group_by_sum")]
    GroupBySum,
    IndexAlignUnion,
    IndexHasDuplicates,
    IndexFirstPositions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureJoinType {
    Inner,
    Left,
}

impl FixtureJoinType {
    #[must_use]
    pub fn into_join_type(self) -> JoinType {
        match self {
            Self::Inner => JoinType::Inner,
            Self::Left => JoinType::Left,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureOracleSource {
    Fixture,
    LiveLegacyPandas,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureSeries {
    pub name: String,
    pub index: Vec<IndexLabel>,
    pub values: Vec<Scalar>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureExpectedSeries {
    pub index: Vec<IndexLabel>,
    pub values: Vec<Scalar>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureExpectedAlignment {
    pub union_index: Vec<IndexLabel>,
    pub left_positions: Vec<Option<usize>>,
    pub right_positions: Vec<Option<usize>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureExpectedJoin {
    pub index: Vec<IndexLabel>,
    pub left_values: Vec<Scalar>,
    pub right_values: Vec<Scalar>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PacketFixture {
    pub packet_id: String,
    pub case_id: String,
    pub mode: RuntimeMode,
    pub operation: FixtureOperation,
    #[serde(default)]
    pub oracle_source: Option<FixtureOracleSource>,
    #[serde(default)]
    pub left: Option<FixtureSeries>,
    #[serde(default)]
    pub right: Option<FixtureSeries>,
    #[serde(default)]
    pub index: Option<Vec<IndexLabel>>,
    #[serde(default)]
    pub join_type: Option<FixtureJoinType>,
    #[serde(default)]
    pub expected_series: Option<FixtureExpectedSeries>,
    #[serde(default)]
    pub expected_join: Option<FixtureExpectedJoin>,
    #[serde(default)]
    pub expected_alignment: Option<FixtureExpectedAlignment>,
    #[serde(default)]
    pub expected_bool: Option<bool>,
    #[serde(default)]
    pub expected_positions: Option<Vec<Option<usize>>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseResult {
    pub packet_id: String,
    pub case_id: String,
    pub mode: RuntimeMode,
    pub operation: FixtureOperation,
    pub status: CaseStatus,
    pub mismatch: Option<String>,
    pub evidence_records: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PacketParityReport {
    pub suite: String,
    pub packet_id: Option<String>,
    pub oracle_present: bool,
    pub fixture_count: usize,
    pub passed: usize,
    pub failed: usize,
    pub results: Vec<CaseResult>,
}

impl PacketParityReport {
    #[must_use]
    pub fn is_green(&self) -> bool {
        self.failed == 0 && self.fixture_count > 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PacketGateResult {
    pub packet_id: String,
    pub pass: bool,
    pub fixture_count: usize,
    pub strict_total: usize,
    pub strict_failed: usize,
    pub hardened_total: usize,
    pub hardened_failed: usize,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RaptorQPacketRecord {
    pub source_block_number: u8,
    pub encoding_symbol_id: u32,
    pub is_source: bool,
    pub serialized_hex: String,
    pub symbol_hash: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RaptorQScrubReport {
    pub verified_at_unix_ms: u64,
    pub status: String,
    pub packet_count: usize,
    pub invalid_packets: usize,
    pub source_hash_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RaptorQSidecarArtifact {
    #[serde(flatten)]
    pub envelope: RaptorQEnvelope,
    pub oti_serialized_hex: String,
    pub source_packets: usize,
    pub repair_packets: usize,
    pub repair_packets_per_block: u32,
    pub packet_records: Vec<RaptorQPacketRecord>,
    pub scrub_report: RaptorQScrubReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WrittenPacketArtifacts {
    pub packet_id: String,
    pub parity_report_path: PathBuf,
    pub raptorq_sidecar_path: PathBuf,
    pub decode_proof_path: PathBuf,
    pub gate_result_path: PathBuf,
    pub mismatch_corpus_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PacketDriftHistoryEntry {
    pub ts_unix_ms: u64,
    pub packet_id: String,
    pub suite: String,
    pub fixture_count: usize,
    pub passed: usize,
    pub failed: usize,
    pub strict_failed: usize,
    pub hardened_failed: usize,
    pub gate_pass: bool,
    pub report_hash: String,
}

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),
    #[error(transparent)]
    Frame(#[from] FrameError),
    #[error("fixture format error: {0}")]
    FixtureFormat(String),
    #[error("oracle is unavailable: {0}")]
    OracleUnavailable(String),
    #[error("oracle command failed: status={status}, stderr={stderr}")]
    OracleCommandFailed { status: i32, stderr: String },
    #[error("raptorq error: {0}")]
    RaptorQ(String),
}

#[derive(Debug, Deserialize)]
struct ParityGateConfig {
    packet_id: String,
    strict: StrictGateConfig,
    hardened: HardenedGateConfig,
    machine_check: MachineCheckConfig,
}

#[derive(Debug, Deserialize)]
struct StrictGateConfig {
    critical_drift_budget: usize,
    non_critical_drift_budget_percent: f64,
}

#[derive(Debug, Deserialize)]
struct HardenedGateConfig {
    divergence_budget_percent: f64,
    allowlisted_divergence_categories: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct MachineCheckConfig {
    suite: String,
    require_fixture_count_at_least: usize,
    require_failed: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct OracleRequest {
    operation: FixtureOperation,
    left: Option<FixtureSeries>,
    right: Option<FixtureSeries>,
    index: Option<Vec<IndexLabel>>,
    join_type: Option<FixtureJoinType>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OracleResponse {
    #[serde(default)]
    expected_series: Option<FixtureExpectedSeries>,
    #[serde(default)]
    expected_join: Option<FixtureExpectedJoin>,
    #[serde(default)]
    expected_alignment: Option<FixtureExpectedAlignment>,
    #[serde(default)]
    expected_bool: Option<bool>,
    #[serde(default)]
    expected_positions: Option<Vec<Option<usize>>>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum ResolvedExpected {
    Series(FixtureExpectedSeries),
    Join(FixtureExpectedJoin),
    Alignment(FixtureExpectedAlignment),
    Bool(bool),
    Positions(Vec<Option<usize>>),
}

pub fn run_packet_suite(config: &HarnessConfig) -> Result<PacketParityReport, HarnessError> {
    run_packet_suite_with_options(config, &SuiteOptions::default())
}

pub fn run_packet_suite_with_options(
    config: &HarnessConfig,
    options: &SuiteOptions,
) -> Result<PacketParityReport, HarnessError> {
    let fixtures = load_fixtures(config, options.packet_filter.as_deref())?;
    build_report(
        config,
        "phase2c_packets".to_owned(),
        None,
        &fixtures,
        options,
    )
}

pub fn run_packet_by_id(
    config: &HarnessConfig,
    packet_id: &str,
    oracle_mode: OracleMode,
) -> Result<PacketParityReport, HarnessError> {
    let options = SuiteOptions {
        packet_filter: Some(packet_id.to_owned()),
        oracle_mode,
    };
    let fixtures = load_fixtures(config, Some(packet_id))?;
    build_report(
        config,
        format!("phase2c_packets:{packet_id}"),
        Some(packet_id.to_owned()),
        &fixtures,
        &options,
    )
}

pub fn run_packets_grouped(
    config: &HarnessConfig,
    options: &SuiteOptions,
) -> Result<Vec<PacketParityReport>, HarnessError> {
    let fixtures = load_fixtures(config, options.packet_filter.as_deref())?;
    let mut grouped = BTreeMap::<String, Vec<PacketFixture>>::new();
    for fixture in fixtures {
        grouped
            .entry(fixture.packet_id.clone())
            .or_default()
            .push(fixture);
    }

    let mut reports = Vec::with_capacity(grouped.len());
    for (packet_id, mut packet_fixtures) in grouped {
        packet_fixtures.sort_by(|a, b| a.case_id.cmp(&b.case_id));
        reports.push(build_report(
            config,
            format!("phase2c_packets:{packet_id}"),
            Some(packet_id),
            &packet_fixtures,
            options,
        )?);
    }
    Ok(reports)
}

pub fn write_grouped_artifacts(
    config: &HarnessConfig,
    reports: &[PacketParityReport],
) -> Result<Vec<WrittenPacketArtifacts>, HarnessError> {
    reports
        .iter()
        .map(|report| write_packet_artifacts(config, report))
        .collect()
}

pub fn enforce_packet_gates(
    config: &HarnessConfig,
    reports: &[PacketParityReport],
) -> Result<(), HarnessError> {
    let mut failures = Vec::new();
    for report in reports {
        let packet_id = report.packet_id.as_deref().unwrap_or("<unknown>");
        if !report.is_green() {
            failures.push(format!(
                "{packet_id}: parity report failed fixtures={}",
                report.failed
            ));
        }
        let gate = evaluate_parity_gate(config, report)?;
        if !gate.pass {
            failures.push(format!(
                "{packet_id}: gate failed reasons={}",
                gate.reasons.join("; ")
            ));
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(HarnessError::FixtureFormat(format!(
            "phase2c enforcement failed: {}",
            failures.join(" | ")
        )))
    }
}

pub fn append_phase2c_drift_history(
    config: &HarnessConfig,
    reports: &[PacketParityReport],
) -> Result<PathBuf, HarnessError> {
    let history_path = config
        .repo_root
        .join("artifacts/phase2c/drift_history.jsonl");
    if let Some(parent) = history_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&history_path)?;

    for report in reports {
        let gate = evaluate_parity_gate(config, report)?;
        let report_json = serde_json::to_vec(report)?;
        let entry = PacketDriftHistoryEntry {
            ts_unix_ms: now_unix_ms(),
            packet_id: report
                .packet_id
                .clone()
                .unwrap_or_else(|| "<unknown>".to_owned()),
            suite: report.suite.clone(),
            fixture_count: report.fixture_count,
            passed: report.passed,
            failed: report.failed,
            strict_failed: gate.strict_failed,
            hardened_failed: gate.hardened_failed,
            gate_pass: gate.pass,
            report_hash: format!("sha256:{}", hash_bytes(&report_json)),
        };
        writeln!(file, "{}", serde_json::to_string(&entry)?)?;
    }

    Ok(history_path)
}

pub fn write_packet_artifacts(
    config: &HarnessConfig,
    report: &PacketParityReport,
) -> Result<WrittenPacketArtifacts, HarnessError> {
    let packet_id = report
        .packet_id
        .as_deref()
        .ok_or_else(|| HarnessError::FixtureFormat("packet_id is required".to_owned()))?;

    let root = config.packet_artifact_root(packet_id);
    fs::create_dir_all(&root)?;

    let parity_report_path = root.join("parity_report.json");
    fs::write(&parity_report_path, serde_json::to_string_pretty(report)?)?;

    let report_bytes = fs::read(&parity_report_path)?;
    let mut sidecar = generate_raptorq_sidecar(
        &format!("{packet_id}/parity_report"),
        "conformance",
        &report_bytes,
        8,
    )?;
    let decode_proof = run_raptorq_decode_recovery_drill(&sidecar, &report_bytes)?;
    sidecar.envelope.decode_proofs = vec![decode_proof.clone()];
    sidecar.envelope.scrub = ScrubStatus {
        last_ok_unix_ms: sidecar.scrub_report.verified_at_unix_ms,
        status: if sidecar.scrub_report.source_hash_verified {
            "ok".to_owned()
        } else {
            "failed".to_owned()
        },
    };

    let raptorq_sidecar_path = root.join("parity_report.raptorq.json");
    fs::write(
        &raptorq_sidecar_path,
        serde_json::to_string_pretty(&sidecar)?,
    )?;

    let decode_proof_path = root.join("parity_report.decode_proof.json");
    let decode_payload = serde_json::json!({
        "packet_id": packet_id,
        "decode_proofs": [decode_proof],
        "status": "recovered",
    });
    fs::write(
        &decode_proof_path,
        serde_json::to_string_pretty(&decode_payload)?,
    )?;

    let gate_result = evaluate_parity_gate(config, report)?;
    let gate_result_path = root.join("parity_gate_result.json");
    fs::write(
        &gate_result_path,
        serde_json::to_string_pretty(&gate_result)?,
    )?;

    let mismatch_corpus_path = root.join("parity_mismatch_corpus.json");
    let mismatches = report
        .results
        .iter()
        .filter(|result| matches!(result.status, CaseStatus::Fail))
        .cloned()
        .collect::<Vec<_>>();
    let mismatch_payload = serde_json::json!({
        "packet_id": packet_id,
        "mismatch_count": mismatches.len(),
        "mismatches": mismatches,
    });
    fs::write(
        &mismatch_corpus_path,
        serde_json::to_string_pretty(&mismatch_payload)?,
    )?;

    Ok(WrittenPacketArtifacts {
        packet_id: packet_id.to_owned(),
        parity_report_path,
        raptorq_sidecar_path,
        decode_proof_path,
        gate_result_path,
        mismatch_corpus_path,
    })
}

pub fn evaluate_parity_gate(
    config: &HarnessConfig,
    report: &PacketParityReport,
) -> Result<PacketGateResult, HarnessError> {
    let packet_id = report
        .packet_id
        .clone()
        .ok_or_else(|| HarnessError::FixtureFormat("report has no packet_id".to_owned()))?;
    let gate: ParityGateConfig =
        serde_yaml::from_str(&fs::read_to_string(config.parity_gate_path(&packet_id))?)?;

    let strict_total = report
        .results
        .iter()
        .filter(|result| matches!(result.mode, RuntimeMode::Strict))
        .count();
    let strict_failed = report
        .results
        .iter()
        .filter(|result| {
            matches!(result.mode, RuntimeMode::Strict) && matches!(result.status, CaseStatus::Fail)
        })
        .count();
    let hardened_total = report
        .results
        .iter()
        .filter(|result| matches!(result.mode, RuntimeMode::Hardened))
        .count();
    let hardened_failed = report
        .results
        .iter()
        .filter(|result| {
            matches!(result.mode, RuntimeMode::Hardened)
                && matches!(result.status, CaseStatus::Fail)
        })
        .count();

    let strict_failure_percent = percent(strict_failed, strict_total);
    let hardened_failure_percent = percent(hardened_failed, hardened_total);

    let mut reasons = Vec::new();
    if gate.packet_id != packet_id {
        reasons.push(format!(
            "packet_id mismatch between gate ({}) and report ({packet_id})",
            gate.packet_id
        ));
    }
    if gate.machine_check.suite != "phase2c_packets"
        && gate.machine_check.suite != report.suite
        && !report.suite.starts_with(&gate.machine_check.suite)
    {
        reasons.push(format!(
            "suite mismatch: gate={}, report={}",
            gate.machine_check.suite, report.suite
        ));
    }
    if report.fixture_count < gate.machine_check.require_fixture_count_at_least {
        reasons.push(format!(
            "fixture_count={} below required {}",
            report.fixture_count, gate.machine_check.require_fixture_count_at_least
        ));
    }
    if report.failed != gate.machine_check.require_failed {
        reasons.push(format!(
            "failed={} but gate requires {}",
            report.failed, gate.machine_check.require_failed
        ));
    }
    if strict_failed > gate.strict.critical_drift_budget {
        reasons.push(format!(
            "strict_failed={} exceeds critical_drift_budget={}",
            strict_failed, gate.strict.critical_drift_budget
        ));
    }
    if strict_failure_percent > gate.strict.non_critical_drift_budget_percent {
        reasons.push(format!(
            "strict failure percent {:.3}% exceeds {:.3}%",
            strict_failure_percent, gate.strict.non_critical_drift_budget_percent
        ));
    }
    if hardened_failure_percent > gate.hardened.divergence_budget_percent {
        reasons.push(format!(
            "hardened failure percent {:.3}% exceeds {:.3}%",
            hardened_failure_percent, gate.hardened.divergence_budget_percent
        ));
    }
    if let Some(categories) = &gate.hardened.allowlisted_divergence_categories
        && categories.is_empty()
    {
        reasons.push("hardened allowlist categories must not be empty".to_owned());
    }

    Ok(PacketGateResult {
        packet_id,
        pass: reasons.is_empty(),
        fixture_count: report.fixture_count,
        strict_total,
        strict_failed,
        hardened_total,
        hardened_failed,
        reasons,
    })
}

pub fn generate_raptorq_sidecar(
    artifact_id: &str,
    artifact_type: &str,
    report_bytes: &[u8],
    repair_packets_per_block: u32,
) -> Result<RaptorQSidecarArtifact, HarnessError> {
    if report_bytes.is_empty() {
        return Err(HarnessError::RaptorQ(
            "cannot generate sidecar for empty payload".to_owned(),
        ));
    }

    let encoder = Encoder::with_defaults(report_bytes, 1400);
    let config = encoder.get_config();

    let mut packet_records = Vec::new();
    let mut symbol_hashes = Vec::new();
    let mut source_packets = 0usize;

    for block in encoder.get_block_encoders() {
        for packet in block.source_packets() {
            source_packets += 1;
            let record = packet_record(packet, true);
            symbol_hashes.push(record.symbol_hash.clone());
            packet_records.push(record);
        }
        for packet in block.repair_packets(0, repair_packets_per_block) {
            let record = packet_record(packet, false);
            symbol_hashes.push(record.symbol_hash.clone());
            packet_records.push(record);
        }
    }

    let repair_packets = packet_records.len().saturating_sub(source_packets);
    let source_hash = hash_bytes(report_bytes);
    let mut scrub_report = verify_raptorq_sidecar_internal(
        report_bytes,
        &source_hash,
        &packet_records,
        now_unix_ms(),
    )?;
    scrub_report.status = if scrub_report.invalid_packets == 0 && scrub_report.source_hash_verified
    {
        "ok".to_owned()
    } else {
        "failed".to_owned()
    };

    let envelope = RaptorQEnvelope {
        artifact_id: artifact_id.to_owned(),
        artifact_type: artifact_type.to_owned(),
        source_hash: format!("sha256:{source_hash}"),
        raptorq: RaptorQMetadata {
            k: source_packets as u32,
            repair_symbols: repair_packets as u32,
            overhead_ratio: if source_packets == 0 {
                0.0
            } else {
                repair_packets as f64 / source_packets as f64
            },
            symbol_hashes,
        },
        scrub: ScrubStatus {
            last_ok_unix_ms: scrub_report.verified_at_unix_ms,
            status: scrub_report.status.clone(),
        },
        decode_proofs: Vec::new(),
    };

    Ok(RaptorQSidecarArtifact {
        envelope,
        oti_serialized_hex: hex_encode(&config.serialize()),
        source_packets,
        repair_packets,
        repair_packets_per_block,
        packet_records,
        scrub_report,
    })
}

pub fn run_raptorq_decode_recovery_drill(
    sidecar: &RaptorQSidecarArtifact,
    report_bytes: &[u8],
) -> Result<DecodeProof, HarnessError> {
    if sidecar.packet_records.is_empty() {
        return Err(HarnessError::RaptorQ(
            "sidecar has no packet records".to_owned(),
        ));
    }

    let oti_bytes = hex_decode(&sidecar.oti_serialized_hex)?;
    if oti_bytes.len() != 12 {
        return Err(HarnessError::RaptorQ(format!(
            "invalid OTI byte length: {}",
            oti_bytes.len()
        )));
    }
    let mut oti = [0_u8; 12];
    oti.copy_from_slice(&oti_bytes);
    let config = ObjectTransmissionInformation::deserialize(&oti);

    let drop_count = sidecar.source_packets.saturating_div(4).max(1);
    let mut dropped_sources = 0usize;
    let mut packets = Vec::with_capacity(sidecar.packet_records.len());
    for record in &sidecar.packet_records {
        if record.is_source && dropped_sources < drop_count {
            dropped_sources += 1;
            continue;
        }

        let packet_bytes = hex_decode(&record.serialized_hex)?;
        packets.push(EncodingPacket::deserialize(&packet_bytes));
    }

    let mut decoder = Decoder::new(config);
    let mut recovered = None;
    for packet in packets {
        recovered = decoder.decode(packet);
        if recovered.is_some() {
            break;
        }
    }

    let recovered = recovered.ok_or_else(|| {
        HarnessError::RaptorQ("decode drill could not reconstruct payload".to_owned())
    })?;
    if recovered != report_bytes {
        return Err(HarnessError::RaptorQ(
            "decode drill recovered bytes do not match source payload".to_owned(),
        ));
    }

    let proof_material = format!(
        "{}:{}:{}",
        sidecar.envelope.artifact_id,
        dropped_sources,
        hash_bytes(&recovered)
    );

    Ok(DecodeProof {
        ts_unix_ms: now_unix_ms(),
        reason: format!(
            "raptorq decode drill dropped {dropped_sources} source packets and recovered payload"
        ),
        recovered_blocks: dropped_sources as u32,
        proof_hash: format!("sha256:{}", hash_bytes(proof_material.as_bytes())),
    })
}

pub fn verify_raptorq_sidecar(
    sidecar: &RaptorQSidecarArtifact,
    report_bytes: &[u8],
) -> Result<RaptorQScrubReport, HarnessError> {
    let expected = sidecar
        .envelope
        .source_hash
        .strip_prefix("sha256:")
        .ok_or_else(|| {
            HarnessError::RaptorQ("source hash must be prefixed with sha256:".to_owned())
        })?
        .to_owned();
    verify_raptorq_sidecar_internal(
        report_bytes,
        &expected,
        &sidecar.packet_records,
        now_unix_ms(),
    )
}

fn verify_raptorq_sidecar_internal(
    report_bytes: &[u8],
    expected_source_hash: &str,
    records: &[RaptorQPacketRecord],
    ts_unix_ms: u64,
) -> Result<RaptorQScrubReport, HarnessError> {
    let source_hash_verified = hash_bytes(report_bytes) == expected_source_hash;
    let mut invalid_packets = 0usize;
    for record in records {
        let bytes = hex_decode(&record.serialized_hex)?;
        if hash_bytes(&bytes) != record.symbol_hash {
            invalid_packets += 1;
        }
    }

    Ok(RaptorQScrubReport {
        verified_at_unix_ms: ts_unix_ms,
        status: if source_hash_verified && invalid_packets == 0 {
            "ok".to_owned()
        } else {
            "failed".to_owned()
        },
        packet_count: records.len(),
        invalid_packets,
        source_hash_verified,
    })
}

fn packet_record(packet: EncodingPacket, is_source: bool) -> RaptorQPacketRecord {
    let payload = packet.payload_id();
    let serialized = packet.serialize();
    RaptorQPacketRecord {
        source_block_number: payload.source_block_number(),
        encoding_symbol_id: payload.encoding_symbol_id(),
        is_source,
        serialized_hex: hex_encode(&serialized),
        symbol_hash: hash_bytes(&serialized),
    }
}

fn build_report(
    config: &HarnessConfig,
    suite: String,
    packet_id: Option<String>,
    fixtures: &[PacketFixture],
    options: &SuiteOptions,
) -> Result<PacketParityReport, HarnessError> {
    let mut results = Vec::with_capacity(fixtures.len());
    for fixture in fixtures {
        results.push(run_fixture(config, fixture, options)?);
    }

    let failed = results
        .iter()
        .filter(|result| matches!(result.status, CaseStatus::Fail))
        .count();
    let passed = results.len().saturating_sub(failed);

    Ok(PacketParityReport {
        suite,
        packet_id,
        oracle_present: config.oracle_root.exists(),
        fixture_count: results.len(),
        passed,
        failed,
        results,
    })
}

fn load_fixtures(
    config: &HarnessConfig,
    packet_filter: Option<&str>,
) -> Result<Vec<PacketFixture>, HarnessError> {
    let fixture_files = list_fixture_files(&config.packet_fixture_root())?;
    let mut fixtures = Vec::with_capacity(fixture_files.len());

    for fixture_path in fixture_files {
        let fixture = load_fixture(&fixture_path)?;
        if packet_filter.is_none_or(|packet| fixture.packet_id == packet) {
            fixtures.push(fixture);
        }
    }
    fixtures.sort_by(|a, b| a.case_id.cmp(&b.case_id));
    Ok(fixtures)
}

fn load_fixture(path: &Path) -> Result<PacketFixture, HarnessError> {
    let body = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&body)?)
}

fn list_fixture_files(root: &Path) -> Result<Vec<PathBuf>, HarnessError> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(current) = stack.pop() {
        for entry in fs::read_dir(current)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "json") {
                files.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

fn run_fixture(
    config: &HarnessConfig,
    fixture: &PacketFixture,
    options: &SuiteOptions,
) -> Result<CaseResult, HarnessError> {
    let policy = match fixture.mode {
        RuntimeMode::Strict => RuntimePolicy::strict(),
        RuntimeMode::Hardened => RuntimePolicy::hardened(Some(100_000)),
    };

    let mut ledger = EvidenceLedger::new();
    let mismatch =
        run_fixture_operation(config, fixture, &policy, &mut ledger, options.oracle_mode).err();

    Ok(CaseResult {
        packet_id: fixture.packet_id.clone(),
        case_id: fixture.case_id.clone(),
        mode: fixture.mode,
        operation: fixture.operation,
        status: if mismatch.is_none() {
            CaseStatus::Pass
        } else {
            CaseStatus::Fail
        },
        mismatch,
        evidence_records: ledger.records().len(),
    })
}

fn run_fixture_operation(
    config: &HarnessConfig,
    fixture: &PacketFixture,
    policy: &RuntimePolicy,
    ledger: &mut EvidenceLedger,
    default_oracle_mode: OracleMode,
) -> Result<(), String> {
    let expected = resolve_expected(config, fixture, default_oracle_mode)
        .map_err(|err| format!("expected resolution failed: {err}"))?;

    match fixture.operation {
        FixtureOperation::SeriesAdd => {
            let left = require_left_series(fixture)?;
            let right = require_right_series(fixture)?;
            let actual = build_series(left)?
                .add_with_policy(&build_series(right)?, policy, ledger)
                .map_err(|err| err.to_string())?;

            let expected = match expected {
                ResolvedExpected::Series(series) => series,
                _ => return Err("expected_series is required for series_add".to_owned()),
            };
            compare_series_expected(&actual, &expected)
        }
        FixtureOperation::SeriesJoin => {
            let left = require_left_series(fixture)?;
            let right = require_right_series(fixture)?;
            let join_type = require_join_type(fixture)?;
            let joined = join_series(
                &build_series(left).map_err(|err| format!("left series build failed: {err}"))?,
                &build_series(right).map_err(|err| format!("right series build failed: {err}"))?,
                join_type.into_join_type(),
            )
            .map_err(|err| err.to_string())?;

            let expected = match expected {
                ResolvedExpected::Join(join) => join,
                _ => return Err("expected_join is required for series_join".to_owned()),
            };
            compare_join_expected(&joined, &expected)
        }
        FixtureOperation::GroupBySum => {
            let keys = require_left_series(fixture)?;
            let values = require_right_series(fixture)?;
            let actual = groupby_sum(
                &build_series(keys).map_err(|err| format!("keys series build failed: {err}"))?,
                &build_series(values)
                    .map_err(|err| format!("values series build failed: {err}"))?,
                GroupByOptions::default(),
                policy,
                ledger,
            )
            .map_err(|err| err.to_string())?;

            let expected = match expected {
                ResolvedExpected::Series(series) => series,
                _ => return Err("expected_series is required for groupby_sum".to_owned()),
            };
            compare_series_expected(&actual, &expected)
        }
        FixtureOperation::IndexAlignUnion => {
            let left = require_left_series(fixture)?;
            let right = require_right_series(fixture)?;
            let plan = align_union(
                &Index::new(left.index.clone()),
                &Index::new(right.index.clone()),
            );
            validate_alignment_plan(&plan).map_err(|err| err.to_string())?;

            let expected = match expected {
                ResolvedExpected::Alignment(alignment) => alignment,
                _ => return Err("expected_alignment is required for index_align_union".to_owned()),
            };
            compare_alignment_expected(&plan, &expected)
        }
        FixtureOperation::IndexHasDuplicates => {
            let index = require_index(fixture)?;
            let actual = Index::new(index.clone()).has_duplicates();
            let expected = match expected {
                ResolvedExpected::Bool(value) => value,
                _ => return Err("expected_bool is required for index_has_duplicates".to_owned()),
            };
            if actual != expected {
                return Err(format!(
                    "duplicate mismatch: actual={actual}, expected={expected}"
                ));
            }
            Ok(())
        }
        FixtureOperation::IndexFirstPositions => {
            let index = require_index(fixture)?;
            let index = Index::new(index.clone());
            let positions = index.position_map_first();
            let actual = index
                .labels()
                .iter()
                .map(|label| positions.get(label).copied())
                .collect::<Vec<_>>();
            let expected = match expected {
                ResolvedExpected::Positions(values) => values,
                _ => {
                    return Err(
                        "expected_positions is required for index_first_positions".to_owned()
                    );
                }
            };
            if actual != expected {
                return Err(format!(
                    "first-position mismatch: actual={actual:?}, expected={expected:?}"
                ));
            }
            Ok(())
        }
    }
}

fn resolve_expected(
    config: &HarnessConfig,
    fixture: &PacketFixture,
    default_mode: OracleMode,
) -> Result<ResolvedExpected, HarnessError> {
    let requested_mode = fixture
        .oracle_source
        .map(|source| match source {
            FixtureOracleSource::Fixture => OracleMode::FixtureExpected,
            FixtureOracleSource::LiveLegacyPandas => OracleMode::LiveLegacyPandas,
        })
        .unwrap_or(default_mode);

    match requested_mode {
        OracleMode::FixtureExpected => fixture_expected(fixture),
        OracleMode::LiveLegacyPandas => capture_live_oracle_expected(config, fixture),
    }
}

fn fixture_expected(fixture: &PacketFixture) -> Result<ResolvedExpected, HarnessError> {
    match fixture.operation {
        FixtureOperation::SeriesAdd => fixture
            .expected_series
            .clone()
            .map(ResolvedExpected::Series)
            .ok_or_else(|| {
                HarnessError::FixtureFormat(format!(
                    "missing expected_series for case {}",
                    fixture.case_id
                ))
            }),
        FixtureOperation::SeriesJoin => fixture
            .expected_join
            .clone()
            .map(ResolvedExpected::Join)
            .ok_or_else(|| {
                HarnessError::FixtureFormat(format!(
                    "missing expected_join for case {}",
                    fixture.case_id
                ))
            }),
        FixtureOperation::GroupBySum => fixture
            .expected_series
            .clone()
            .map(ResolvedExpected::Series)
            .ok_or_else(|| {
                HarnessError::FixtureFormat(format!(
                    "missing expected_series for case {}",
                    fixture.case_id
                ))
            }),
        FixtureOperation::IndexAlignUnion => fixture
            .expected_alignment
            .clone()
            .map(ResolvedExpected::Alignment)
            .ok_or_else(|| {
                HarnessError::FixtureFormat(format!(
                    "missing expected_alignment for case {}",
                    fixture.case_id
                ))
            }),
        FixtureOperation::IndexHasDuplicates => fixture
            .expected_bool
            .map(ResolvedExpected::Bool)
            .ok_or_else(|| {
                HarnessError::FixtureFormat(format!(
                    "missing expected_bool for case {}",
                    fixture.case_id
                ))
            }),
        FixtureOperation::IndexFirstPositions => fixture
            .expected_positions
            .clone()
            .map(ResolvedExpected::Positions)
            .ok_or_else(|| {
                HarnessError::FixtureFormat(format!(
                    "missing expected_positions for case {}",
                    fixture.case_id
                ))
            }),
    }
}

fn capture_live_oracle_expected(
    config: &HarnessConfig,
    fixture: &PacketFixture,
) -> Result<ResolvedExpected, HarnessError> {
    if !config.oracle_root.exists() {
        return Err(HarnessError::OracleUnavailable(format!(
            "legacy oracle root does not exist: {}",
            config.oracle_root.display()
        )));
    }
    let script = config.oracle_script_path();
    if !script.exists() {
        return Err(HarnessError::OracleUnavailable(format!(
            "oracle script does not exist: {}",
            script.display()
        )));
    }

    let payload = OracleRequest {
        operation: fixture.operation,
        left: fixture.left.clone(),
        right: fixture.right.clone(),
        index: fixture.index.clone(),
        join_type: fixture.join_type,
    };
    let input = serde_json::to_vec(&payload)?;

    let output = Command::new(&config.python_bin)
        .arg(&script)
        .arg("--legacy-root")
        .arg(&config.oracle_root)
        .arg("--strict-legacy")
        .args(
            config
                .allow_system_pandas_fallback
                .then_some("--allow-system-pandas-fallback"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(&input)?;
            }
            child.wait_with_output()
        })?;

    if !output.status.success() {
        if let Ok(response) = serde_json::from_slice::<OracleResponse>(&output.stdout)
            && let Some(error) = response.error
        {
            return Err(HarnessError::OracleUnavailable(error));
        }
        let code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        return Err(HarnessError::OracleCommandFailed {
            status: code,
            stderr: format!("{stderr}\nstdout={stdout}"),
        });
    }

    let response: OracleResponse = serde_json::from_slice(&output.stdout)?;
    if let Some(error) = response.error {
        return Err(HarnessError::OracleUnavailable(error));
    }

    match fixture.operation {
        FixtureOperation::SeriesAdd => response
            .expected_series
            .map(ResolvedExpected::Series)
            .ok_or_else(|| {
                HarnessError::FixtureFormat("oracle omitted expected_series".to_owned())
            }),
        FixtureOperation::SeriesJoin => response
            .expected_join
            .map(ResolvedExpected::Join)
            .ok_or_else(|| HarnessError::FixtureFormat("oracle omitted expected_join".to_owned())),
        FixtureOperation::GroupBySum => response
            .expected_series
            .map(ResolvedExpected::Series)
            .ok_or_else(|| {
                HarnessError::FixtureFormat("oracle omitted expected_series".to_owned())
            }),
        FixtureOperation::IndexAlignUnion => response
            .expected_alignment
            .map(ResolvedExpected::Alignment)
            .ok_or_else(|| {
                HarnessError::FixtureFormat("oracle omitted expected_alignment".to_owned())
            }),
        FixtureOperation::IndexHasDuplicates => response
            .expected_bool
            .map(ResolvedExpected::Bool)
            .ok_or_else(|| HarnessError::FixtureFormat("oracle omitted expected_bool".to_owned())),
        FixtureOperation::IndexFirstPositions => response
            .expected_positions
            .map(ResolvedExpected::Positions)
            .ok_or_else(|| {
                HarnessError::FixtureFormat("oracle omitted expected_positions".to_owned())
            }),
    }
}

fn require_left_series(fixture: &PacketFixture) -> Result<&FixtureSeries, String> {
    fixture
        .left
        .as_ref()
        .ok_or_else(|| "missing left fixture series".to_owned())
}

fn require_right_series(fixture: &PacketFixture) -> Result<&FixtureSeries, String> {
    fixture
        .right
        .as_ref()
        .ok_or_else(|| "missing right fixture series".to_owned())
}

fn require_index(fixture: &PacketFixture) -> Result<&Vec<IndexLabel>, String> {
    fixture
        .index
        .as_ref()
        .ok_or_else(|| "missing index fixture vector".to_owned())
}

fn require_join_type(fixture: &PacketFixture) -> Result<FixtureJoinType, String> {
    fixture
        .join_type
        .ok_or_else(|| "missing join_type for join fixture".to_owned())
}

fn build_series(series: &FixtureSeries) -> Result<Series, String> {
    Series::from_values(
        series.name.clone(),
        series.index.clone(),
        series.values.clone(),
    )
    .map_err(|err| err.to_string())
}

fn compare_series_expected(
    actual: &Series,
    expected: &FixtureExpectedSeries,
) -> Result<(), String> {
    if actual.index().labels() != expected.index {
        return Err(format!(
            "index mismatch: actual={:?}, expected={:?}",
            actual.index().labels(),
            expected.index
        ));
    }

    if actual.values().len() != expected.values.len() {
        return Err(format!(
            "value length mismatch: actual={}, expected={}",
            actual.values().len(),
            expected.values.len()
        ));
    }

    for (idx, (left, right)) in actual
        .values()
        .iter()
        .zip(expected.values.iter())
        .enumerate()
    {
        if !left.semantic_eq(right) {
            return Err(format!(
                "value mismatch at idx={idx}: actual={left:?}, expected={right:?}"
            ));
        }
    }
    Ok(())
}

fn compare_join_expected(
    actual: &fp_join::JoinedSeries,
    expected: &FixtureExpectedJoin,
) -> Result<(), String> {
    if actual.index.labels() != expected.index {
        return Err(format!(
            "join index mismatch: actual={:?}, expected={:?}",
            actual.index.labels(),
            expected.index
        ));
    }

    if actual.left_values.values().len() != expected.left_values.len() {
        return Err(format!(
            "join left length mismatch: actual={}, expected={}",
            actual.left_values.values().len(),
            expected.left_values.len()
        ));
    }
    if actual.right_values.values().len() != expected.right_values.len() {
        return Err(format!(
            "join right length mismatch: actual={}, expected={}",
            actual.right_values.values().len(),
            expected.right_values.len()
        ));
    }

    for (idx, (left, right)) in actual
        .left_values
        .values()
        .iter()
        .zip(expected.left_values.iter())
        .enumerate()
    {
        let equal = left.semantic_eq(right) || (left.is_missing() && right.is_missing());
        if !equal {
            return Err(format!(
                "join left mismatch at idx={idx}: actual={left:?}, expected={right:?}"
            ));
        }
    }
    for (idx, (left, right)) in actual
        .right_values
        .values()
        .iter()
        .zip(expected.right_values.iter())
        .enumerate()
    {
        let equal = left.semantic_eq(right) || (left.is_missing() && right.is_missing());
        if !equal {
            return Err(format!(
                "join right mismatch at idx={idx}: actual={left:?}, expected={right:?}"
            ));
        }
    }

    Ok(())
}

fn compare_alignment_expected(
    actual: &AlignmentPlan,
    expected: &FixtureExpectedAlignment,
) -> Result<(), String> {
    if actual.union_index.labels() != expected.union_index {
        return Err(format!(
            "union_index mismatch: actual={:?}, expected={:?}",
            actual.union_index.labels(),
            expected.union_index
        ));
    }
    if actual.left_positions != expected.left_positions {
        return Err(format!(
            "left_positions mismatch: actual={:?}, expected={:?}",
            actual.left_positions, expected.left_positions
        ));
    }
    if actual.right_positions != expected.right_positions {
        return Err(format!(
            "right_positions mismatch: actual={:?}, expected={:?}",
            actual.right_positions, expected.right_positions
        ));
    }
    Ok(())
}

fn percent(failed: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (failed as f64 / total as f64) * 100.0
    }
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(hex_digit(byte >> 4));
        out.push(hex_digit(byte & 0x0f));
    }
    out
}

fn hex_decode(value: &str) -> Result<Vec<u8>, HarnessError> {
    if !value.len().is_multiple_of(2) {
        return Err(HarnessError::RaptorQ(format!(
            "invalid hex length {}",
            value.len()
        )));
    }
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(value.len() / 2);
    for idx in (0..bytes.len()).step_by(2) {
        let high = hex_value(bytes[idx])?;
        let low = hex_value(bytes[idx + 1])?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

fn hex_digit(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        10..=15 => (b'a' + (value - 10)) as char,
        _ => unreachable!("nibble out of range"),
    }
}

fn hex_value(byte: u8) -> Result<u8, HarnessError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(HarnessError::RaptorQ(format!(
            "invalid hex character: {}",
            byte as char
        ))),
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{
        CaseStatus, FixtureExpectedAlignment, HarnessConfig, OracleMode, RaptorQSidecarArtifact,
        SuiteOptions, append_phase2c_drift_history, enforce_packet_gates, evaluate_parity_gate,
        generate_raptorq_sidecar, run_packet_by_id, run_packet_suite,
        run_packet_suite_with_options, run_packets_grouped, run_raptorq_decode_recovery_drill,
        run_smoke,
    };

    #[test]
    fn smoke_harness_finds_oracle_and_fixtures() {
        let cfg = HarnessConfig::default_paths();
        let report = run_smoke(&cfg);
        assert!(report.oracle_present, "oracle repo should be present");
        assert!(report.fixture_count >= 1, "expected at least one fixture");
        assert!(report.strict_mode);
    }

    #[test]
    fn packet_suite_is_green_for_bootstrap_cases() {
        let cfg = HarnessConfig::default_paths();
        let report = run_packet_suite(&cfg).expect("suite should run");
        assert!(report.fixture_count >= 1, "expected packet fixtures");
        assert!(report.is_green(), "expected report green: {report:?}");
    }

    #[test]
    fn packet_filter_runs_only_requested_packet() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_packet_by_id(&cfg, "FP-P2C-002", OracleMode::FixtureExpected).expect("report");
        assert_eq!(report.packet_id.as_deref(), Some("FP-P2C-002"));
        assert!(
            report.fixture_count >= 3,
            "expected dedicated FP-P2C-002 fixtures"
        );
        assert!(report.is_green());
    }

    #[test]
    fn packet_filter_runs_join_packet() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_packet_by_id(&cfg, "FP-P2C-004", OracleMode::FixtureExpected).expect("report");
        assert_eq!(report.packet_id.as_deref(), Some("FP-P2C-004"));
        assert!(report.fixture_count >= 3, "expected join packet fixtures");
        assert!(report.is_green());
    }

    #[test]
    fn packet_filter_runs_groupby_packet() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_packet_by_id(&cfg, "FP-P2C-005", OracleMode::FixtureExpected).expect("report");
        assert_eq!(report.packet_id.as_deref(), Some("FP-P2C-005"));
        assert!(
            report.fixture_count >= 3,
            "expected groupby packet fixtures"
        );
        assert!(report.is_green());
    }

    #[test]
    fn grouped_reports_are_partitioned_per_packet() {
        let cfg = HarnessConfig::default_paths();
        let reports = run_packets_grouped(&cfg, &SuiteOptions::default()).expect("grouped");
        assert!(
            reports
                .iter()
                .any(|report| report.packet_id.as_deref() == Some("FP-P2C-001"))
        );
        assert!(
            reports
                .iter()
                .any(|report| report.packet_id.as_deref() == Some("FP-P2C-002"))
        );
        assert!(
            reports
                .iter()
                .any(|report| report.packet_id.as_deref() == Some("FP-P2C-003"))
        );
        assert!(
            reports
                .iter()
                .any(|report| report.packet_id.as_deref() == Some("FP-P2C-004"))
        );
        assert!(
            reports
                .iter()
                .any(|report| report.packet_id.as_deref() == Some("FP-P2C-005"))
        );
        enforce_packet_gates(&cfg, &reports).expect("enforcement should pass");
    }

    #[test]
    fn packet_gate_enforcement_fails_when_report_is_not_green() {
        let cfg = HarnessConfig::default_paths();
        let mut reports = run_packets_grouped(&cfg, &SuiteOptions::default()).expect("grouped");
        let report = reports.first_mut().expect("at least one packet");
        let first_case = report.results.first_mut().expect("at least one case");
        first_case.status = CaseStatus::Fail;
        first_case.mismatch = Some("synthetic non-green check".to_owned());
        report.failed = 1;
        report.passed = report.fixture_count.saturating_sub(1);

        let err = enforce_packet_gates(&cfg, &reports).expect_err("should fail");
        let message = err.to_string();
        assert!(
            message.contains("enforcement failed"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn drift_history_append_emits_jsonl_rows() {
        let cfg = HarnessConfig::default_paths();
        let reports = run_packets_grouped(&cfg, &SuiteOptions::default()).expect("grouped");
        let history_path = append_phase2c_drift_history(&cfg, &reports).expect("history");
        let contents = fs::read_to_string(&history_path).expect("history content");
        let latest = contents.lines().last().expect("at least one row");
        let row: serde_json::Value = serde_json::from_str(latest).expect("json row");
        assert!(
            row.get("packet_id").is_some(),
            "history row should include packet_id"
        );
        assert!(
            row.get("gate_pass").is_some(),
            "history row should include gate pass status"
        );
    }

    #[test]
    fn parity_gate_evaluation_passes_for_packet_001() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_packet_by_id(&cfg, "FP-P2C-001", OracleMode::FixtureExpected).expect("report");
        let result = evaluate_parity_gate(&cfg, &report).expect("gate");
        assert!(result.pass, "gate should pass: {result:?}");
    }

    #[test]
    fn parity_gate_evaluation_fails_for_injected_drift() {
        let cfg = HarnessConfig::default_paths();
        let mut report =
            run_packet_by_id(&cfg, "FP-P2C-001", OracleMode::FixtureExpected).expect("report");
        let first = report.results.first_mut().expect("at least one result");
        first.status = CaseStatus::Fail;
        first.mismatch = Some("synthetic drift injection".to_owned());
        report.failed = 1;
        report.passed = report.fixture_count.saturating_sub(1);

        let result = evaluate_parity_gate(&cfg, &report).expect("gate");
        assert!(!result.pass, "gate should fail for injected drift");
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.contains("failed="))
        );
    }

    #[test]
    fn raptorq_sidecar_round_trip_recovery_drill_passes() {
        let payload = br#"{\"suite\":\"phase2c_packets\",\"passed\":4,\"failed\":0}"#;
        let sidecar = generate_raptorq_sidecar("test/parity_report", "conformance", payload, 8)
            .expect("sidecar generation");
        let proof = run_raptorq_decode_recovery_drill(&sidecar, payload).expect("decode drill");
        assert!(proof.recovered_blocks >= 1);
    }

    #[test]
    fn index_alignment_expected_type_serialization_is_stable() {
        let expected = FixtureExpectedAlignment {
            union_index: vec![1_i64.into(), 2_i64.into()],
            left_positions: vec![Some(0), None],
            right_positions: vec![None, Some(0)],
        };
        let json = serde_json::to_string(&expected).expect("serialize");
        let back: FixtureExpectedAlignment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, expected);
    }

    #[test]
    fn live_oracle_mode_executes_or_returns_structured_failure() {
        let cfg = HarnessConfig::default_paths();
        let options = SuiteOptions {
            packet_filter: Some("FP-P2C-001".to_owned()),
            oracle_mode: OracleMode::LiveLegacyPandas,
        };
        let result = run_packet_suite_with_options(&cfg, &options);
        match result {
            Ok(report) => assert!(report.fixture_count >= 1),
            Err(err) => {
                let message = err.to_string();
                assert!(
                    message.contains("oracle"),
                    "expected oracle-class error, got {message}"
                );
            }
        }
    }

    #[test]
    fn sidecar_verification_runs_on_generated_artifact() {
        let payload = br#"{\"suite\":\"phase2c_packets\",\"passed\":2,\"failed\":0}"#;
        let sidecar: RaptorQSidecarArtifact =
            generate_raptorq_sidecar("test/artifact", "conformance", payload, 8).expect("sidecar");
        let scrub = super::verify_raptorq_sidecar(&sidecar, payload).expect("scrub");
        assert_eq!(scrub.status, "ok");
    }
}
