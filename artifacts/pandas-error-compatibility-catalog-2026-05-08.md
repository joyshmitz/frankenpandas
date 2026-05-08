# Pandas Error Compatibility Catalog - 2026-05-08

Bead: `br-frankenpandas-tn6qb.7`

Scope: strict-mode pandas-observable exception compatibility. The catalog is
for representative error families where callers commonly branch on failure
class, operation path, or a stable message fragment. Hardened-mode recovery is
out of scope and must use separate policy tests.

## Catalog

| Family | Representative API | Pandas-observable behavior | FrankenPandas harness target | Pinning policy |
| --- | --- | --- | --- | --- |
| Dtype misuse | `astype`, safe dtype coercions, invalid `errors=` values | Raises a dtype or argument-class exception with the bad dtype/argument visible | `CompatibilityRejected` or typed frame/series error at the operation boundary | Pin the error family and stable bad dtype/argument fragment. Avoid exact full messages unless pandas has made the wording contractual for that path. |
| Missing labels or columns | `DataFrame.pivot(values="missing")`, label-based selection | Pandas `KeyError` with the missing label visible; local snapshot: `KeyError 'missing'` | `DataFramePivot` expected-error fixture with `column 'missing' not found` fragment | Pin the missing-label category and exact label fragment. Do not require pandas punctuation when FrankenPandas has an explicit compatibility wrapper. |
| Duplicate labels or keys | `DataFrame.pivot(index=..., columns=...)` with duplicate output cells | Pandas `ValueError` with duplicate-entry reshape text; local snapshot: `Index contains duplicate entries, cannot reshape` | `DataFramePivot` expected-error fixture with `duplicate entries` fragment | Pin `ValueError` family in oracle captures and a stable duplicate-entry fragment in fixture-backed tests. |
| Malformed IO | CSV/JSONL with malformed records, bad delimiters, truncated strings | Parser-specific exception class and offending syntax/location where stable | IO fixture expected-error packets under strict parser mode | Pin parser category, input path, and shortest stable syntax fragment. Hardened recovery reports belong in hardened-only tests. |
| Unsupported feature or parameter | Unknown pandas keyword, unsupported engine/mode, unsupported aggregation token | Pandas rejects fail-closed before producing partial output | `CompatibilityRejected` at argument normalization or operation dispatch | Pin feature or parameter name plus fail-closed status. Do not turn unsupported features into silent fallbacks. |
| Strict versus hardened policy | Same malformed or ambiguous input under strict and hardened runtime modes | Strict preserves pandas-observable rejection; hardened may recover with bounded audit trail | Separate strict and hardened fixture packets | Never reuse a hardened recovery expectation as strict parity evidence. |

## Fixture Packets

The first harness slice uses fixture-backed expected errors instead of relying
on a live pandas subprocess being available on every worker:

| Packet | Case | Operation | Expected fragment |
| --- | --- | --- | --- |
| `FP-ERR-CATALOG-001` | `dataframe_pivot_duplicate_keys_expected_error_contains_tn6qb7` | `dataframe_pivot` | `duplicate entries` |
| `FP-ERR-CATALOG-002` | `dataframe_pivot_missing_value_column_expected_error_contains_tn6qb7` | `dataframe_pivot` | `column 'missing' not found` |

Both cases run through `run_differential_fixture` with
`OracleMode::FixtureExpected`, which proves the existing differential harness
can mark expected pandas-style error paths as passing only when the operation
actually fails with the required stable fragment and produces no drift records.

## Guidance

- Prefer live-oracle captures when the worker has pandas available, but store a
  fixture-backed `expected_error_contains` packet for durable CI evidence.
- Exact message equality is allowed only for stable, documented, or already
  fixture-proven fragments. Otherwise pin class/category, operation path, and
  the shortest meaningful substring.
- A skipped live oracle is not parity proof. Treat it as unavailable evidence
  unless a fixture-backed expected error packet also covers the case.
- Missing-label and duplicate-key failures are high-value families because user
  code often catches these errors while reshaping or validating schemas.
- When adding new families, include one strict fixture packet, one note about
  whether pandas class parity is required, and one short fragment that survives
  harmless wording changes.
