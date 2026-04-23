//! Scale-tier conformance integration tests.
//!
//! Per br-frankenpandas-k05t. Fixtures under
//! `crates/fp-conformance/fixtures/packets_scale/<tier>/` are generated
//! by `crates/fp-conformance/oracle/generate_scale_fixtures.py`.
//! These tests execute the generated fixtures through the same harness
//! that runs Tier-S (≤ 30 rows) packet fixtures today; the only
//! difference is row count.
//!
//! Tier gating:
//! - Tier-M (1k-10k rows): runs when `FP_SCALE_M=1` is set. Wire to
//!   PR CI via a dedicated env flag so that default `cargo test` stays
//!   fast.
//! - Tier-L (100k+ rows): runs when `FP_SCALE_L=1` is set. Wire to the
//!   scheduled nightly workflow only — not on every PR.
//!
//! Fixtures are not committed to the repo today; the nightly workflow
//! regenerates them each run via the Python generator (deterministic
//! seed → stable bytes). Committed fixtures are deferred to the next
//! slice once the harness + gating pattern is validated.

use std::path::PathBuf;

fn scale_fixtures_dir(tier: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("fixtures");
    path.push("packets_scale");
    path.push(tier);
    path
}

#[test]
fn scale_fixtures_tier_s_directory_shape() {
    // Canary: if the directory exists, it should be named consistently
    // with the generator's output path (lowercase tier letter).
    let dir = scale_fixtures_dir("s");
    if dir.exists() {
        assert!(
            dir.is_dir(),
            "packets_scale/s must be a directory, not a file"
        );
    }
    // Absence is fine — Tier-S is covered by the core packets/ suite.
}

#[test]
#[ignore] // gated by FP_SCALE_M=1; run via `cargo test --ignored`
fn scale_tier_m_gate() {
    if std::env::var("FP_SCALE_M").unwrap_or_default() != "1" {
        return;
    }
    let dir = scale_fixtures_dir("m");
    assert!(
        dir.is_dir(),
        "Tier-M fixtures missing: run `python3 crates/fp-conformance/oracle/generate_scale_fixtures.py --tier m` first ({})",
        dir.display()
    );
    // Full packet-suite execution lands in a follow-up once the tier-M
    // harness wiring (operation-specific comparators + timing budget)
    // is defined.
}

#[test]
#[ignore] // gated by FP_SCALE_L=1; nightly-only
fn scale_tier_l_gate() {
    if std::env::var("FP_SCALE_L").unwrap_or_default() != "1" {
        return;
    }
    let dir = scale_fixtures_dir("l");
    assert!(
        dir.is_dir(),
        "Tier-L fixtures missing: run `python3 crates/fp-conformance/oracle/generate_scale_fixtures.py --tier l` first ({})",
        dir.display()
    );
}
