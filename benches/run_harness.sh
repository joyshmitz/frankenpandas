#!/usr/bin/env bash
# run_harness.sh - Run vs-pandas head-to-head benchmark harness
#
# Usage:
#   ./benches/run_harness.sh --all --sizes 10k,100k
#   ./benches/run_harness.sh --category io --sizes 100k
#
# Requires: Python 3.10+, pandas, numpy
# Optional: fp-bench binary (built with release-perf profile)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info() { echo "[$(date -u +%H:%M:%S)] INFO: $*"; }
log_warn() { echo "[$(date -u +%H:%M:%S)] WARN: $*" >&2; }

cd "$PROJECT_ROOT"

# Check Python dependencies
if ! python3 -c "import pandas, numpy" 2>/dev/null; then
    log_warn "Missing Python dependencies. Install with:"
    echo "  pip install pandas numpy"
    exit 1
fi

# Build FrankenPandas bench binary if not present
FP_BENCH="$PROJECT_ROOT/target/release-perf/fp-bench"
if [[ ! -f "$FP_BENCH" ]]; then
    log_info "Building fp-bench with release-perf profile..."
    if cargo build --profile release-perf -p fp-bench 2>/dev/null; then
        log_info "fp-bench built successfully"
    else
        log_warn "fp-bench not available (fp-bench crate may not exist yet)"
        log_warn "Proceeding with pandas-only benchmarks"
    fi
fi

log_info "Running vs-pandas benchmark harness..."
python3 "$SCRIPT_DIR/vs_pandas_harness.py" "$@"
