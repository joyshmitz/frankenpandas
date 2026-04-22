#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "$repo_root" <<'PY'
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys

repo_root = pathlib.Path(sys.argv[1])
requirements_path = repo_root / "crates/fp-conformance/oracle/requirements.txt"
oracle_script_path = repo_root / "crates/fp-conformance/oracle/pandas_oracle.py"
fixture_root = repo_root / "crates/fp-conformance/fixtures/packets"

requirements = requirements_path.read_text(encoding="utf-8")
match = re.search(r"(?m)^\s*pandas==([^\s#]+)\s*$", requirements)
if not match:
    print(
        f"fixture freshness check failed: missing pandas== pin in {requirements_path}",
        file=sys.stderr,
    )
    raise SystemExit(1)

expected_pandas = match.group(1)
expected_sha256 = hashlib.sha256(oracle_script_path.read_bytes()).hexdigest()
errors: list[str] = []
checked = 0

for fixture_path in sorted(fixture_root.glob("*.json")):
    checked += 1
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    provenance = fixture.get("fixture_provenance")
    if not isinstance(provenance, dict):
        errors.append(f"{fixture_path}: missing fixture_provenance object")
        continue

    pandas_version = provenance.get("pandas_version")
    if pandas_version != expected_pandas:
        errors.append(
            f"{fixture_path}: pandas_version={pandas_version!r} expected {expected_pandas!r}"
        )

    oracle_sha256 = provenance.get("oracle_script_sha256")
    if oracle_sha256 != expected_sha256:
        errors.append(
            f"{fixture_path}: oracle_script_sha256={oracle_sha256!r} expected {expected_sha256!r}"
        )

    generated_at = provenance.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        errors.append(f"{fixture_path}: generated_at missing or blank")
        continue

    try:
        dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{fixture_path}: generated_at is not RFC3339/ISO8601: {generated_at!r}")

if errors:
    print(
        f"fixture freshness check failed: {len(errors)} issue(s) across {checked} fixtures",
        file=sys.stderr,
    )
    for error in errors[:50]:
        print(f"  - {error}", file=sys.stderr)
    if len(errors) > 50:
        print(f"  - ... {len(errors) - 50} more", file=sys.stderr)
    raise SystemExit(1)

print(
    f"fixture freshness OK: {checked} fixtures pinned to pandas {expected_pandas} and oracle {expected_sha256}"
)
PY
