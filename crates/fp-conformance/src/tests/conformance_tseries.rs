//! pandas.tseries parity-matrix conformance suite (frankenpandas-dh4f).
//!
//! Per /testing-conformance-harnesses Pattern 1, these tests compare the
//! Rust `fp_index::date_range` facade against live upstream pandas
//! `pd.date_range` for edge-case parameter combinations.

use std::{
    io::Write,
    process::{Command, Stdio},
};

use fp_index::{Index, IndexLabel, date_range};
use fp_types::Timedelta;
use serde::Deserialize;

use super::HarnessConfig;

#[derive(Debug, Clone, Copy)]
struct DateRangeCase<'a> {
    case_id: &'a str,
    start: Option<&'a str>,
    end: Option<&'a str>,
    periods: Option<usize>,
    pandas_freq: &'a str,
    rust_freq_nanos: i64,
    name: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct PandasDateRange {
    values: Vec<i64>,
    name: Option<String>,
}

fn pandas_date_range_or_skip(
    config: &HarnessConfig,
    case: DateRangeCase<'_>,
) -> Result<Option<PandasDateRange>, String> {
    if !config.oracle_root.exists() && !config.allows_live_oracle_fallback() {
        eprintln!(
            "live pandas unavailable; skipping tseries conformance test {}: legacy oracle root does not exist: {}",
            case.case_id,
            config.oracle_root.display()
        );
        return Ok(None);
    }

    let payload = serde_json::json!({
        "start": case.start,
        "end": case.end,
        "periods": case.periods,
        "freq": case.pandas_freq,
        "name": case.name,
    });

    let script = r#"
import importlib
import json
import os
import sys

legacy_root = os.path.abspath(sys.argv[1])
allow_system_fallback = sys.argv[2] == "1"
candidate_parent = os.path.dirname(legacy_root)
if os.path.isdir(candidate_parent):
    sys.path.insert(0, candidate_parent)

try:
    import pandas as pd
except Exception:
    if not allow_system_fallback:
        raise
    while candidate_parent in sys.path:
        sys.path.remove(candidate_parent)
    sys.modules.pop("pandas", None)
    pd = importlib.import_module("pandas")

request = json.loads(sys.stdin.read())
kwargs = {"freq": request["freq"]}
for key in ("start", "end", "periods", "name"):
    if request.get(key) is not None:
        kwargs[key] = request[key]

index = pd.date_range(**kwargs)
print(json.dumps({
    "values": [int(value) for value in index.asi8.tolist()],
    "name": index.name,
}, separators=(",", ":")))
"#;

    let mut child = Command::new(&config.python_bin)
        .arg("-c")
        .arg(script)
        .arg(&config.oracle_root)
        .arg(if config.allows_live_oracle_fallback() {
            "1"
        } else {
            "0"
        })
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("spawn pandas date_range oracle failed: {err}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "pandas date_range oracle stdin unavailable".to_owned())?;
    stdin
        .write_all(payload.to_string().as_bytes())
        .map_err(|err| format!("write pandas date_range oracle payload failed: {err}"))?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .map_err(|err| format!("wait for pandas date_range oracle failed: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "pandas date_range oracle failed for {}: {}",
            case.case_id,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    serde_json::from_slice(&output.stdout)
        .map(Some)
        .map_err(|err| format!("parse pandas date_range json failed: {err}"))
}

fn datetime_labels(index: &Index) -> Result<Vec<i64>, String> {
    index
        .labels()
        .iter()
        .map(|label| match label {
            IndexLabel::Datetime64(value) => Ok(*value),
            other => Err(format!("expected datetime64 label, got {other:?}")),
        })
        .collect()
}

fn assert_date_range_matches_pandas(case: DateRangeCase<'_>) -> Result<(), String> {
    let config = HarnessConfig::default_paths();
    let Some(expected) = pandas_date_range_or_skip(&config, case)? else {
        return Ok(());
    };

    let actual = date_range(
        case.start,
        case.end,
        case.periods,
        case.rust_freq_nanos,
        case.name,
    )
    .map_err(|err| format!("fp date_range failed for {}: {err}", case.case_id))?;

    assert_eq!(
        datetime_labels(&actual)?,
        expected.values,
        "pandas.tseries date_range value parity drift for {}",
        case.case_id
    );
    assert_eq!(
        actual.name().map(str::to_owned),
        expected.name,
        "pandas.tseries date_range name parity drift for {}",
        case.case_id
    );
    Ok(())
}

#[test]
fn conformance_tseries_date_range_start_periods_daily() -> Result<(), String> {
    assert_date_range_matches_pandas(DateRangeCase {
        case_id: "tseries_date_range_start_periods_daily",
        start: Some("2024-01-01"),
        end: None,
        periods: Some(3),
        pandas_freq: "D",
        rust_freq_nanos: Timedelta::NANOS_PER_DAY,
        name: None,
    })
}

#[test]
fn conformance_tseries_date_range_start_end_two_day_frequency() -> Result<(), String> {
    assert_date_range_matches_pandas(DateRangeCase {
        case_id: "tseries_date_range_start_end_two_day_frequency",
        start: Some("2024-01-01"),
        end: Some("2024-01-07"),
        periods: None,
        pandas_freq: "2D",
        rust_freq_nanos: 2 * Timedelta::NANOS_PER_DAY,
        name: None,
    })
}

#[test]
fn conformance_tseries_date_range_end_periods_hourly() -> Result<(), String> {
    assert_date_range_matches_pandas(DateRangeCase {
        case_id: "tseries_date_range_end_periods_hourly",
        start: None,
        end: Some("2024-01-01 12:00:00"),
        periods: Some(3),
        pandas_freq: "6h",
        rust_freq_nanos: 6 * Timedelta::NANOS_PER_HOUR,
        name: None,
    })
}

#[test]
fn conformance_tseries_date_range_zero_periods() -> Result<(), String> {
    assert_date_range_matches_pandas(DateRangeCase {
        case_id: "tseries_date_range_zero_periods",
        start: Some("2024-01-01"),
        end: None,
        periods: Some(0),
        pandas_freq: "D",
        rust_freq_nanos: Timedelta::NANOS_PER_DAY,
        name: None,
    })
}

#[test]
fn conformance_tseries_date_range_preserves_name() -> Result<(), String> {
    assert_date_range_matches_pandas(DateRangeCase {
        case_id: "tseries_date_range_preserves_name",
        start: Some("2024-02-01"),
        end: None,
        periods: Some(2),
        pandas_freq: "D",
        rust_freq_nanos: Timedelta::NANOS_PER_DAY,
        name: Some("when"),
    })
}
