# Conformance Fixtures

This folder stores normalized oracle-vs-target fixtures for fp-conformance.

## Structure

- `smoke_case.json`: minimal bootstrap fixture ensuring harness wiring works.
- `packets/*.json`: Phase-2C packet fixtures consumed by `run_packet_suite`.

## Provenance

**Generator:** `../oracle/pandas_oracle.py`
**Reference implementation:** pandas (system install, typically 2.x)
**Generation workflow:**
```bash
# Generate fixture from oracle
python3 ../oracle/pandas_oracle.py \
  --legacy-root ../../legacy_pandas_code/pandas \
  < request.json > response.json
```

## Fixture Format

Each packet fixture is a JSON object with:
- `packet_id`: Unique identifier (e.g., "FP-P2C-001")
- `case_id`: Human-readable test case name
- `mode`: "strict" or "hardened"
- `operation`: The pandas operation being tested
- `left`, `right`, `frame`: Input data
- `expected_*`: Expected output from pandas oracle

## Regenerating Fixtures

When pandas behavior changes or new operations are added:

1. Update the oracle script to handle new operations
2. Generate new fixtures via the oracle
3. Review diff: `git diff fixtures/`
4. Document any divergences in `../DISCREPANCIES.md`
5. Commit with message referencing the change

## Current Coverage

See `../COVERAGE.md` for detailed coverage matrix.
