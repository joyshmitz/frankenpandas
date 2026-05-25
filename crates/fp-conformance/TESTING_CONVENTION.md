# FrankenPandas Testing Convention

Every fix bead must satisfy the following checklist before close.

## Required Test Coverage

### 1. Unit Tests (inline `#[cfg(test)]`)

Each fix must include unit tests covering:

- **Happy path**: Basic functionality with valid inputs
- **Edge cases**:
  - Empty inputs (empty Series, empty DataFrame, zero rows)
  - All-null columns
  - NaN values in numeric columns
  - Inf/-Inf values
  - Boundary values (min/max integers, date bounds)
  - Unicode strings (emoji, CJK, RTL)
- **Error semantics**: If pandas raises an error for certain inputs, FrankenPandas must raise an equivalent error (test with `#[should_panic]` or `Result::Err` assertions)

### 2. Conformance Packet

At least one differential conformance packet must be added:

- Location: `fixtures/packets/FP_P2D_NNN/`
- Include `fixture_provenance.json` documenting:
  - pandas version used (target: 2.2.3)
  - Generation date
  - Input parameters
  - Expected output (captured from live pandas)

### 3. Pre-Commit Verification

Before closing a fix bead:

```bash
# Run conformance delta script (no red-flips allowed)
./scripts/conformance-delta.sh

# Check changed files with ubs
ubs <changed-files>
# Must exit 0

# Clippy must be clean
cargo clippy -D warnings
```

## Critical Rule: Verify Against LIVE Pandas

**BEFORE** deciding that the subject (FrankenPandas) has a bug:

1. Run the operation against **live pandas 2.2.3**
2. Compare expected vs actual output
3. If the fixture disagrees with live pandas, the **fixture is wrong**, not FrankenPandas
4. Regenerate the fixture from live pandas output, do not "fix" FrankenPandas to match a stale fixture

This rule exists because many fixtures were generated with older pandas versions or incorrect input parameters.

## Test File Naming

- Unit tests: Add to existing `#[cfg(test)] mod tests` in the affected crate's `lib.rs`
- Test function naming: `<operation>_<scenario>` (e.g., `combine_first_fills_nulls_from_other`)
- Conformance packets: `FP_P2D_NNN_<short_description>.json`

## Checklist Template

Copy this to fix bead bodies:

```
## Pre-Close Checklist

- [ ] Unit tests: happy path
- [ ] Unit tests: edge cases (empty, null, NaN, inf, boundaries)
- [ ] Unit tests: error semantics match pandas
- [ ] Conformance packet with fixture_provenance
- [ ] Verified against live pandas (not just fixture)
- [ ] conformance-delta.sh: no red-flips
- [ ] ubs <files>: exit 0
- [ ] cargo clippy -D warnings: clean
```
