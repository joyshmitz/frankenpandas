#!/usr/bin/env bash
# install-hooks.sh — point git at the committed ./githooks directory.
#
# Run once after every fresh clone. Idempotent; safe to re-run.
# Tracked under br-frankenpandas-pa2y.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
if [ -z "${repo_root}" ]; then
    echo "error: not inside a git repo" >&2
    exit 1
fi

hooks_dir="${repo_root}/githooks"
if [ ! -d "${hooks_dir}" ]; then
    echo "error: ${hooks_dir} does not exist" >&2
    exit 1
fi

git config core.hooksPath "${hooks_dir}"

echo "✓ core.hooksPath set to ${hooks_dir}"
echo "  All future git commit / git push operations will run hooks from that directory."
echo ""
echo "  To verify: git config --get core.hooksPath"
echo "  To disable: git config --unset core.hooksPath"
