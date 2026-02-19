#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# If rch is available, offload the expensive cargo build/run to the worker fleet.
# Opt out with FP_NO_RCH=1 (useful for CI or debugging local toolchain issues).
if [[ "${FP_NO_RCH:-}" != "1" ]] && command -v rch >/dev/null 2>&1; then
  echo "info: offloading phase2c gate check via rch (set FP_NO_RCH=1 to disable)" >&2
  exec rch exec -- env FP_NO_RCH=1 bash scripts/phase2c_gate_check.sh
fi

mkdir -p .tmp .target
export TMPDIR="$repo_root/.tmp"
export CARGO_TARGET_DIR="$repo_root/.target"

cargo run -p fp-conformance --bin fp-conformance-cli -- \
  --oracle fixture \
  --write-artifacts \
  --require-green
