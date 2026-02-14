# ROUND4 Isomorphism Proof

Change:
- Dense `Int64` aggregation path in `groupby_sum`.

Behavior checks:
- Ordering preserved: yes, first-seen key order tracked explicitly.
- Tie-breaking unchanged: yes, first occurrence still controls output order.
- Floating-point drift: unchanged accumulation model (`f64` sum path retained).
- RNG seeds: N/A.

Validation artifacts:
- `cargo test -p fp-groupby` passed.
- `cargo test -p fp-conformance` passed.
- `cargo run -p fp-conformance --bin fp-conformance-cli -- --write-artifacts --require-green` passed.
- `(cd artifacts/perf && sha256sum -c golden_checksums.txt)` passed.

Observed result:
- Mean improvement was small (`~0.75%`), but no behavioral drift observed.
