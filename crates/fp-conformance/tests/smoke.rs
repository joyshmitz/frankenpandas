use std::path::Path;

use fp_conformance::{HarnessConfig, run_packet_suite, run_smoke};

#[test]
fn smoke_report_is_stable() {
    let cfg = HarnessConfig::default_paths();
    let report = run_smoke(&cfg);
    assert_eq!(report.suite, "smoke");
    assert!(report.fixture_count >= 1);
    assert!(report.oracle_present);

    let fixture_path = cfg.fixture_root.join("smoke_case.json");
    assert!(Path::new(&fixture_path).exists());
}

#[test]
fn packet_suite_executes_at_least_one_case() {
    let cfg = HarnessConfig::default_paths();
    let report = run_packet_suite(&cfg).expect("packet suite should run");
    assert!(report.fixture_count >= 1);
    assert!(report.passed >= 1);
}
