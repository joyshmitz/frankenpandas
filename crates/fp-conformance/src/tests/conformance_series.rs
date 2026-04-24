//! Series parity-matrix conformance suite (br-frankenpandas-3qt8).
//!
//! Per /testing-conformance-harnesses skill Pattern 4 (spec-derived test
//! matrix). Each test picks one Series-family operation and runs it
//! through an edge-case input (empty, single-row, all-NaN, duplicate
//! labels, misaligned indexes). The live pandas oracle is the reference
//! implementation; our Rust result must match via the standard
//! `compare_series_expected` / scalar comparators.
//!
//! Each test skips gracefully (no failure) when the live oracle isn't
//! available — matches the convention of sibling `live_oracle_*` tests.

use super::{
    EvidenceLedger, FixtureExpectedSeries, HarnessConfig, HarnessError, PacketFixture,
    ResolvedExpected, RuntimePolicy, Scalar, build_series, capture_live_oracle_expected,
    compare_scalar, compare_series_expected,
};

fn oracle_series_expected(
    cfg: &HarnessConfig,
    fixture: &PacketFixture,
) -> Result<Option<FixtureExpectedSeries>, String> {
    match capture_live_oracle_expected(cfg, fixture) {
        Ok(ResolvedExpected::Series(series)) => Ok(Some(series)),
        Ok(other) => Err(format!("expected series payload, got {other:?}")),
        Err(HarnessError::OracleUnavailable(message)) => {
            eprintln!(
                "live pandas unavailable; skipping Series conformance test {}: {message}",
                fixture.case_id
            );
            Ok(None)
        }
        Err(err) => Err(format!("oracle error on {}: {err}", fixture.case_id)),
    }
}

fn oracle_scalar_expected(
    cfg: &HarnessConfig,
    fixture: &PacketFixture,
) -> Result<Option<Scalar>, String> {
    match capture_live_oracle_expected(cfg, fixture) {
        Ok(ResolvedExpected::Scalar(scalar)) => Ok(Some(scalar)),
        Ok(other) => Err(format!("expected scalar payload, got {other:?}")),
        Err(HarnessError::OracleUnavailable(message)) => {
            eprintln!(
                "live pandas unavailable; skipping Series conformance test {}: {message}",
                fixture.case_id
            );
            Ok(None)
        }
        Err(err) => Err(format!("oracle error on {}: {err}", fixture.case_id)),
    }
}

fn strict_config() -> HarnessConfig {
    HarnessConfig::default_paths()
}

/// Helper: run series_add against the oracle + compare.
fn check_series_add(fixture: PacketFixture) {
    let cfg = strict_config();
    let Some(expected) = oracle_series_expected(&cfg, &fixture).expect("series oracle") else {
        return;
    };
    let left = build_series(fixture.left.as_ref().expect("left series")).expect("left build");
    let right = build_series(fixture.right.as_ref().expect("right series")).expect("right build");
    let policy = RuntimePolicy::strict();
    let mut ledger = EvidenceLedger::new();
    let actual = left
        .add_with_policy(&right, &policy, &mut ledger)
        .expect("series_add");
    compare_series_expected(&actual, &expected).expect("pandas series_add parity");
}

fn check_series_mode(fixture: PacketFixture) {
    let cfg = strict_config();
    let Some(expected) = oracle_series_expected(&cfg, &fixture).expect("series oracle") else {
        return;
    };
    let series = build_series(fixture.left.as_ref().expect("left series")).expect("series build");
    // Default dropna=true matches pandas Series.mode default.
    let actual = series.mode().expect("series_mode");
    compare_series_expected(&actual, &expected).expect("pandas series_mode parity");
}

fn check_series_nunique(fixture: PacketFixture) {
    let cfg = strict_config();
    let Some(expected) = oracle_scalar_expected(&cfg, &fixture).expect("scalar oracle") else {
        return;
    };
    let series = build_series(fixture.left.as_ref().expect("left series")).expect("series build");
    let actual = Scalar::Int64(series.nunique() as i64);
    compare_scalar(&actual, &expected, "series_nunique").expect("pandas series_nunique parity");
}

// ── series_add edge matrix ────────────────────────────────────────────

#[test]
fn conformance_series_add_empty_pair() {
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-SERIES-ADD-001",
        "case_id": "series_add_empty_pair",
        "mode": "strict",
        "operation": "series_add",
        "oracle_source": "live_legacy_pandas",
        "left":  { "name": "l", "index": [], "values": [] },
        "right": { "name": "r", "index": [], "values": [] }
    }))
    .expect("fixture");
    check_series_add(fixture);
}

#[test]
fn conformance_series_add_single_row() {
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-SERIES-ADD-002",
        "case_id": "series_add_single_row",
        "mode": "strict",
        "operation": "series_add",
        "oracle_source": "live_legacy_pandas",
        "left":  { "name": "l", "index": [{ "kind": "int64", "value": 0 }],
                   "values": [{ "kind": "float64", "value": 42.0 }] },
        "right": { "name": "r", "index": [{ "kind": "int64", "value": 0 }],
                   "values": [{ "kind": "float64", "value": 8.0 }] }
    }))
    .expect("fixture");
    check_series_add(fixture);
}

#[test]
fn conformance_series_add_all_nan() {
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-SERIES-ADD-003",
        "case_id": "series_add_all_nan",
        "mode": "strict",
        "operation": "series_add",
        "oracle_source": "live_legacy_pandas",
        "left":  { "name": "l", "index": [
                       { "kind": "int64", "value": 0 },
                       { "kind": "int64", "value": 1 },
                       { "kind": "int64", "value": 2 }],
                   "values": [
                       { "kind": "null", "value": "na_n" },
                       { "kind": "null", "value": "na_n" },
                       { "kind": "null", "value": "na_n" }] },
        "right": { "name": "r", "index": [
                       { "kind": "int64", "value": 0 },
                       { "kind": "int64", "value": 1 },
                       { "kind": "int64", "value": 2 }],
                   "values": [
                       { "kind": "null", "value": "na_n" },
                       { "kind": "float64", "value": 1.0 },
                       { "kind": "null", "value": "na_n" }] }
    }))
    .expect("fixture");
    check_series_add(fixture);
}

#[test]
fn conformance_series_add_duplicate_labels() {
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-SERIES-ADD-004",
        "case_id": "series_add_duplicate_labels",
        "mode": "strict",
        "operation": "series_add",
        "oracle_source": "live_legacy_pandas",
        "left":  { "name": "l", "index": [
                       { "kind": "int64", "value": 1 },
                       { "kind": "int64", "value": 1 }],
                   "values": [
                       { "kind": "float64", "value": 10.0 },
                       { "kind": "float64", "value": 20.0 }] },
        "right": { "name": "r", "index": [
                       { "kind": "int64", "value": 1 },
                       { "kind": "int64", "value": 1 }],
                   "values": [
                       { "kind": "float64", "value": 100.0 },
                       { "kind": "float64", "value": 200.0 }] }
    }))
    .expect("fixture");
    check_series_add(fixture);
}

#[test]
fn conformance_series_add_misaligned_indexes() {
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-SERIES-ADD-005",
        "case_id": "series_add_misaligned_indexes",
        "mode": "strict",
        "operation": "series_add",
        "oracle_source": "live_legacy_pandas",
        "left":  { "name": "l", "index": [
                       { "kind": "int64", "value": 0 },
                       { "kind": "int64", "value": 1 }],
                   "values": [
                       { "kind": "int64", "value": 10 },
                       { "kind": "int64", "value": 20 }] },
        "right": { "name": "r", "index": [
                       { "kind": "int64", "value": 2 },
                       { "kind": "int64", "value": 3 }],
                   "values": [
                       { "kind": "int64", "value": 100 },
                       { "kind": "int64", "value": 200 }] }
    }))
    .expect("fixture");
    check_series_add(fixture);
}

// ── series_mode edge matrix ───────────────────────────────────────────

#[test]
fn conformance_series_mode_empty() {
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-SERIES-MODE-001",
        "case_id": "series_mode_empty",
        "mode": "strict",
        "operation": "series_mode",
        "oracle_source": "live_legacy_pandas",
        "left": { "name": "s", "index": [], "values": [] }
    }))
    .expect("fixture");
    check_series_mode(fixture);
}

#[test]
fn conformance_series_mode_unique_no_mode() {
    // All values distinct → pandas returns every value (everything ties at 1).
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-SERIES-MODE-002",
        "case_id": "series_mode_unique_no_mode",
        "mode": "strict",
        "operation": "series_mode",
        "oracle_source": "live_legacy_pandas",
        "left": { "name": "s", "index": [
                      { "kind": "int64", "value": 0 },
                      { "kind": "int64", "value": 1 },
                      { "kind": "int64", "value": 2 }],
                  "values": [
                      { "kind": "int64", "value": 1 },
                      { "kind": "int64", "value": 2 },
                      { "kind": "int64", "value": 3 }] }
    }))
    .expect("fixture");
    check_series_mode(fixture);
}

// ── series_nunique edge matrix ───────────────────────────────────────

#[test]
fn conformance_series_nunique_empty() {
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-SERIES-NUNIQUE-001",
        "case_id": "series_nunique_empty",
        "mode": "strict",
        "operation": "series_nunique",
        "oracle_source": "live_legacy_pandas",
        "left": { "name": "s", "index": [], "values": [] }
    }))
    .expect("fixture");
    check_series_nunique(fixture);
}

#[test]
fn conformance_series_nunique_all_nan() {
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-SERIES-NUNIQUE-002",
        "case_id": "series_nunique_all_nan",
        "mode": "strict",
        "operation": "series_nunique",
        "oracle_source": "live_legacy_pandas",
        "left": { "name": "s", "index": [
                      { "kind": "int64", "value": 0 },
                      { "kind": "int64", "value": 1 },
                      { "kind": "int64", "value": 2 }],
                  "values": [
                      { "kind": "null", "value": "na_n" },
                      { "kind": "null", "value": "na_n" },
                      { "kind": "null", "value": "na_n" }] }
    }))
    .expect("fixture");
    check_series_nunique(fixture);
}

#[test]
fn conformance_series_nunique_all_duplicates() {
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-SERIES-NUNIQUE-003",
        "case_id": "series_nunique_all_duplicates",
        "mode": "strict",
        "operation": "series_nunique",
        "oracle_source": "live_legacy_pandas",
        "left": { "name": "s", "index": [
                      { "kind": "int64", "value": 0 },
                      { "kind": "int64", "value": 1 },
                      { "kind": "int64", "value": 2 }],
                  "values": [
                      { "kind": "utf8", "value": "x" },
                      { "kind": "utf8", "value": "x" },
                      { "kind": "utf8", "value": "x" }] }
    }))
    .expect("fixture");
    check_series_nunique(fixture);
}
