//! Reverse-conformance integration tests: pandas reads frankenpandas output.
//!
//! Per br-frankenpandas-kdwn. The forward channel (pandas_oracle.py)
//! validates "frankenpandas answers == pandas answers" for in-memory
//! operations. This channel validates "pandas can parse bytes we
//! write" across the fp-io writer surface. Without it, a drift in
//! our serializer can pass every forward-conformance assert while
//! producing files only our reader accepts.
//!
//! Wire-level flow (once fixtures land):
//!   Rust: build tiny frame → fp_io::write_<fmt> → bytes
//!   Python subprocess: `crates/fp-conformance/oracle/reverse_oracle.py`
//!   Compare: pandas parse against expected schema.
//!
//! This file ships the sentinel test + a unit test that confirms the
//! reverse_oracle.py path exists and is executable. Fixture-driven
//! round-trip coverage lands in a follow-up slice once
//! fixtures/packets_reverse/ is populated.

use std::path::PathBuf;

fn reverse_oracle_script_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("oracle");
    path.push("reverse_oracle.py");
    path
}

#[test]
fn reverse_oracle_script_is_present_and_readable() {
    let script = reverse_oracle_script_path();
    assert!(
        script.is_file(),
        "reverse oracle script missing: {}",
        script.display()
    );
    let src = std::fs::read_to_string(&script).expect("read reverse_oracle.py");
    assert!(
        src.contains("def main()"),
        "reverse_oracle.py must expose a main() entrypoint"
    );
    assert!(
        src.contains("\"parsed_frame\""),
        "reverse_oracle.py must emit parsed_frame key per br-kdwn protocol"
    );
}

#[test]
fn reverse_conformance_module_compiles() {
    // Sentinel test — proves this integration test file is picked up
    // by `cargo test -p fp-conformance --test reverse_conformance` even
    // before fixture-driven cases land. Replace or keep as a canary.
}
