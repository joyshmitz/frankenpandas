//! IO parity-matrix conformance suite (br-frankenpandas-czmt).
//!
//! Per /testing-conformance-harnesses Pattern 1, each test compares
//! FrankenPandas IO behavior with live upstream pandas for edge-case inputs:
//! empty CSVs, single rows, missing-heavy values, quoting, bad-line handling,
//! decimal/boolean parsing options, CSV write/reparse behavior, and JSON
//! records serialization with duplicate index labels.

use super::{
    CaseStatus, HarnessConfig, HarnessError, OracleMode, PacketFixture, ResolvedExpected,
    SuiteOptions, capture_live_oracle_expected,
};

fn strict_config() -> HarnessConfig {
    HarnessConfig::default_paths()
}

fn live_oracle_available(cfg: &HarnessConfig, fixture: &PacketFixture) -> Result<bool, String> {
    match capture_live_oracle_expected(cfg, fixture) {
        Ok(
            ResolvedExpected::Frame(_) | ResolvedExpected::Scalar(_) | ResolvedExpected::Bool(_),
        ) => Ok(true),
        Ok(other) => Err(format!(
            "unexpected live oracle payload for {}: {other:?}",
            fixture.case_id
        )),
        Err(HarnessError::OracleUnavailable(message)) => {
            eprintln!(
                "live pandas unavailable; skipping IO conformance test {}: {message}",
                fixture.case_id
            );
            Ok(false)
        }
        Err(err) => Err(format!("oracle error on {}: {err}", fixture.case_id)),
    }
}

fn check_io_fixture(fixture: PacketFixture) {
    let cfg = strict_config();
    if !live_oracle_available(&cfg, &fixture).expect("IO oracle") {
        return;
    }

    let report = super::run_differential_fixture(
        &cfg,
        &fixture,
        &SuiteOptions {
            packet_filter: None,
            oracle_mode: OracleMode::LiveLegacyPandas,
        },
    )
    .expect("differential report");

    assert_eq!(
        report.status,
        CaseStatus::Pass,
        "pandas IO parity drift for {}: {:?}",
        report.case_id,
        report.drift_records
    );
}

fn csv_fixture(
    packet_id: &str,
    case_id: &str,
    operation: &str,
    csv_input: &str,
    options: &[(&str, serde_json::Value)],
) -> PacketFixture {
    let mut raw = serde_json::json!({
        "packet_id": packet_id,
        "case_id": case_id,
        "mode": "strict",
        "operation": operation,
        "oracle_source": "live_legacy_pandas",
        "csv_input": csv_input
    });
    let raw_object = raw.as_object_mut().expect("fixture object");
    for (key, value) in options {
        raw_object.insert((*key).to_owned(), value.clone());
    }

    serde_json::from_value(raw).expect("fixture")
}

#[test]
fn conformance_io_read_csv_empty_header_only() {
    let fixture = csv_fixture(
        "FP-CONF-IO-001",
        "io_read_csv_empty_header_only",
        "csv_read_frame",
        "a,b\n",
        &[],
    );
    check_io_fixture(fixture);
}

#[test]
fn conformance_io_read_csv_single_row_mixed_dtypes() {
    let fixture = csv_fixture(
        "FP-CONF-IO-002",
        "io_read_csv_single_row_mixed_dtypes",
        "csv_read_frame",
        "id,name,score\n1,Ada,3.5\n",
        &[],
    );
    check_io_fixture(fixture);
}

#[test]
fn conformance_io_read_csv_missing_heavy_values() {
    let fixture = csv_fixture(
        "FP-CONF-IO-003",
        "io_read_csv_missing_heavy_values",
        "csv_read_frame",
        "a,b,c\n,NA,NaN\n,,x\n",
        &[],
    );
    check_io_fixture(fixture);
}

#[test]
fn conformance_io_read_csv_quoted_commas_and_newlines() {
    let fixture = csv_fixture(
        "FP-CONF-IO-004",
        "io_read_csv_quoted_commas_and_newlines",
        "csv_read_frame",
        "name,note\nAda,\"hello, world\"\nBob,\"line1\nline2\"\n",
        &[],
    );
    check_io_fixture(fixture);
}

#[test]
fn conformance_io_read_csv_on_bad_lines_skip() {
    let fixture = csv_fixture(
        "FP-CONF-IO-005",
        "io_read_csv_on_bad_lines_skip",
        "csv_read_frame",
        "a,b\n1,2\n3,4,5\n6,7\n",
        &[("csv_on_bad_lines", serde_json::json!("skip"))],
    );
    check_io_fixture(fixture);
}

#[test]
fn conformance_io_read_csv_true_false_values() {
    let fixture = csv_fixture(
        "FP-CONF-IO-006",
        "io_read_csv_true_false_values",
        "csv_read_frame",
        "flag\nyes\nno\n",
        &[
            ("csv_true_values", serde_json::json!(["yes"])),
            ("csv_false_values", serde_json::json!(["no"])),
        ],
    );
    check_io_fixture(fixture);
}

#[test]
fn conformance_io_read_csv_parse_dates_iso_dates() {
    let fixture = csv_fixture(
        "FP-CONF-IO-007",
        "io_read_csv_parse_dates_iso_dates",
        "csv_read_frame",
        "ts,value\n2024-01-15,1\n2024-01-16,2\n",
        &[("csv_parse_dates", serde_json::json!(["ts"]))],
    );
    check_io_fixture(fixture);
}

#[test]
fn conformance_io_read_csv_parse_dates_combined_columns() {
    let fixture = csv_fixture(
        "FP-CONF-IO-008",
        "io_read_csv_parse_dates_combined_columns",
        "csv_read_frame",
        "date,time,value\n2024-01-15,10:30:00,1\n2024-01-16,11:45:30,2\n",
        &[(
            "csv_parse_date_combinations",
            serde_json::json!([["date", "time"]]),
        )],
    );
    check_io_fixture(fixture);
}

#[test]
fn conformance_io_csv_round_trip_quoted_missing_values() {
    let fixture = csv_fixture(
        "FP-CONF-IO-009",
        "io_csv_round_trip_quoted_missing_values",
        "csv_round_trip",
        "name,note,value\nAda,\"hello, world\",1\nBob,,\n",
        &[],
    );
    check_io_fixture(fixture);
}

#[test]
fn conformance_io_dataframe_to_json_records_ignores_duplicate_index() {
    let fixture: PacketFixture = serde_json::from_value(serde_json::json!({
        "packet_id": "FP-CONF-IO-010",
        "case_id": "io_dataframe_to_json_records_ignores_duplicate_index",
        "mode": "strict",
        "operation": "dataframe_to_json_records",
        "oracle_source": "live_legacy_pandas",
        "frame": {
            "index": [
                { "kind": "utf8", "value": "row" },
                { "kind": "utf8", "value": "row" }
            ],
            "columns": {
                "b": [
                    { "kind": "utf8", "value": "x" },
                    { "kind": "utf8", "value": "y" }
                ],
                "a": [
                    { "kind": "int64", "value": 1 },
                    { "kind": "int64", "value": 2 }
                ],
                "flag": [
                    { "kind": "bool", "value": true },
                    { "kind": "bool", "value": false }
                ]
            },
            "column_order": ["b", "a", "flag"]
        }
    }))
    .expect("fixture");
    check_io_fixture(fixture);
}
