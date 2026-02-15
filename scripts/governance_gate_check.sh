#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p artifacts/ci .tmp .target
export TMPDIR="$repo_root/.tmp"
export CARGO_TARGET_DIR="$repo_root/.target"

cargo run -p fp-conformance --bin fp-governance-gate -- \
  --json-out artifacts/ci/governance_gate_report.json
