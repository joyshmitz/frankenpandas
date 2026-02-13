#![forbid(unsafe_code)]

use fp_conformance::{
    HarnessConfig, OracleMode, SuiteOptions, append_phase2c_drift_history, enforce_packet_gates,
    run_packets_grouped, write_grouped_artifacts,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut packet_filter: Option<String> = None;
    let mut oracle_mode = OracleMode::FixtureExpected;
    let mut write_artifacts = false;
    let mut require_green = false;
    let mut write_drift_history = false;
    let mut allow_system_pandas_fallback = false;

    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--packet-id" => {
                let value = args
                    .next()
                    .ok_or("--packet-id requires a value (e.g. FP-P2C-001)")?;
                packet_filter = Some(value);
            }
            "--oracle" => {
                let value = args.next().ok_or("--oracle requires fixture or live")?;
                oracle_mode = match value.as_str() {
                    "fixture" => OracleMode::FixtureExpected,
                    "live" => OracleMode::LiveLegacyPandas,
                    _ => return Err(format!("unsupported oracle mode: {value}").into()),
                };
            }
            "--write-artifacts" => {
                write_artifacts = true;
            }
            "--require-green" => {
                require_green = true;
            }
            "--write-drift-history" => {
                write_drift_history = true;
            }
            "--allow-system-pandas-fallback" => {
                allow_system_pandas_fallback = true;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
    }

    let mut config = HarnessConfig::default_paths();
    config.allow_system_pandas_fallback = allow_system_pandas_fallback;
    let options = SuiteOptions {
        packet_filter,
        oracle_mode,
    };

    let reports = run_packets_grouped(&config, &options)?;
    for report in &reports {
        println!(
            "packet={} suite={} fixtures={} passed={} failed={} green={}",
            report.packet_id.as_deref().unwrap_or("<all>"),
            report.suite,
            report.fixture_count,
            report.passed,
            report.failed,
            report.is_green()
        );
    }

    if require_green {
        enforce_packet_gates(&config, &reports)?;
    }

    if write_artifacts {
        let written = write_grouped_artifacts(&config, &reports)?;
        for artifact in written {
            println!(
                "wrote packet={} parity={} sidecar={} decode_proof={} gate={} mismatch_corpus={}",
                artifact.packet_id,
                artifact.parity_report_path.display(),
                artifact.raptorq_sidecar_path.display(),
                artifact.decode_proof_path.display(),
                artifact.gate_result_path.display(),
                artifact.mismatch_corpus_path.display()
            );
        }
    }

    if write_artifacts || write_drift_history {
        let history_path = append_phase2c_drift_history(&config, &reports)?;
        println!("wrote drift_history={}", history_path.display());
    }

    Ok(())
}

fn print_help() {
    println!(
        "fp-conformance-cli\n\
         Usage:\n\
         \tfp-conformance-cli [--packet-id FP-P2C-001] [--oracle fixture|live] [--write-artifacts] [--require-green]\n\
         Options:\n\
         \t--packet-id <id>     Run only one packet id\n\
         \t--oracle <mode>      fixture (default) or live\n\
         \t--write-artifacts    Emit parity + gate + RaptorQ sidecars per packet\n\
         \t--write-drift-history Append packet run summary to artifacts/phase2c/drift_history.jsonl\n\
         \t--require-green      Fail with non-zero exit when any packet parity/gate check fails\n\
         \t--allow-system-pandas-fallback  Allow non-legacy pandas import when live oracle is enabled\n\
         \t-h, --help           Show this help"
    );
}
