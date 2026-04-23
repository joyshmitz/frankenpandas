#!/usr/bin/env bash
# Corpus + crash-artifact minimization wrapper for the fp-fuzz crate.
#
# Runs cargo fuzz tmin on every fuzz/artifacts/<target>/crash-* (shrinks
# individual crash reproducers) and cargo fuzz cmin on every fuzz/corpus/
# <target>/ directory (shrinks corpus file count while preserving coverage).
#
# Per br-frankenpandas-i9rj / /testing-fuzzing Rule 8 + Rule 9.
#
# Usage:
#   scripts/fuzz_minimize.sh          # minimize every target
#   scripts/fuzz_minimize.sh <target> # minimize one target
#
# Invoked weekly via .github/workflows/fuzz-minimize.yml.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d fuzz ]; then
  echo "fuzz/ directory not present — nothing to minimize." >&2
  exit 0
fi

cd fuzz

targets_all="$(cargo metadata --no-deps --format-version=1 \
  | jq -r '.packages[] | select(.name == "fp-fuzz") | .targets[] | select(.kind[] == "bin") | .name')"

if [ "$#" -gt 0 ]; then
  targets="$1"
else
  targets="$targets_all"
fi

for target in $targets; do
  echo "=== minimizing target: $target ==="

  # Step 1 — tmin every existing crash artifact (Rule 9).
  artifacts_dir="artifacts/$target"
  if [ -d "$artifacts_dir" ]; then
    for crash in "$artifacts_dir"/crash-* "$artifacts_dir"/oom-* "$artifacts_dir"/timeout-*; do
      [ -f "$crash" ] || continue
      echo "  tmin: $crash"
      cargo +nightly fuzz tmin "$target" "$crash" || true
    done
  fi

  # Step 2 — cmin the persistent corpus (Rule 8).
  corpus_dir="corpus/$target"
  if [ -d "$corpus_dir" ]; then
    echo "  cmin: $corpus_dir"
    cargo +nightly fuzz cmin "$target" || true
  fi
done

echo "=== minimization pass complete ==="
