#!/usr/bin/env bash
# Performance ratchet gate for CI integration
#
# Runs the vs-pandas benchmark harness and compares against committed baseline.
# Exit codes:
#   0 = ALLOW (all pass)
#   1 = BLOCK (regression detected)
#   2 = QUARANTINE (high cv, needs review)
#
# Usage:
#   ./scripts/apply_ratchet.sh                    # Run full benchmark + ratchet
#   ./scripts/apply_ratchet.sh --quick            # Quick subset for PR checks
#   ./scripts/apply_ratchet.sh --update-baseline  # Update baseline after merge

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASELINE_DIR="$PROJECT_ROOT/.bench-history"
ARTIFACTS_DIR="$PROJECT_ROOT/artifacts/bench"

# Ensure directories exist
mkdir -p "$BASELINE_DIR" "$ARTIFACTS_DIR"

# Parse arguments
QUICK_MODE=false
UPDATE_BASELINE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --update-baseline)
            UPDATE_BASELINE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_FILE="$ARTIFACTS_DIR/bench-$TIMESTAMP.json"

echo "=== FrankenPandas Performance Ratchet Gate ==="
echo "Timestamp: $TIMESTAMP"
echo "Output: $OUTPUT_FILE"
echo ""

# Run benchmarks
if [ "$QUICK_MODE" = true ]; then
    echo "Running quick benchmark subset..."
    python3 "$PROJECT_ROOT/benches/vs_pandas_harness.py" \
        --category groupby \
        --sizes 10k \
        --output "$OUTPUT_FILE"
else
    echo "Running full benchmark suite..."
    python3 "$PROJECT_ROOT/benches/vs_pandas_harness.py" \
        --all \
        --sizes 10k,100k \
        --output "$OUTPUT_FILE"
fi

# Check if baseline exists
BASELINE_FILE="$BASELINE_DIR/latest.json"
if [ ! -f "$BASELINE_FILE" ]; then
    echo ""
    echo "No baseline found. Initializing with current results..."
    python3 "$SCRIPT_DIR/perf_ratchet.py" --update-baseline "$OUTPUT_FILE"
    echo "ALLOW: First run, baseline initialized"
    exit 0
fi

# Update baseline mode
if [ "$UPDATE_BASELINE" = true ]; then
    echo ""
    echo "Updating baseline..."
    python3 "$SCRIPT_DIR/perf_ratchet.py" --update-baseline "$OUTPUT_FILE"
    echo "ALLOW: Baseline updated"
    exit 0
fi

# Run ratchet comparison
echo ""
echo "Comparing against baseline..."
REPORT_FILE="$ARTIFACTS_DIR/ratchet-report-$TIMESTAMP.json"
python3 "$SCRIPT_DIR/perf_ratchet.py" \
    --baseline "$BASELINE_FILE" \
    --new "$OUTPUT_FILE" \
    --output "$REPORT_FILE"

EXIT_CODE=$?

# Archive results
if [ -d "$BASELINE_DIR" ]; then
    cp "$OUTPUT_FILE" "$BASELINE_DIR/bench-$TIMESTAMP.json"
fi

exit $EXIT_CODE
