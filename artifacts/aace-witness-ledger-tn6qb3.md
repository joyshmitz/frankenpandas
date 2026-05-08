# AACE Witness Ledger Slice - br-frankenpandas-tn6qb.3

Scope: first narrow semantic witness record for `Series` binary arithmetic alignment.

Witness fields:
- input index identity: role, length, duplicate flag, and stable SHA-256 fingerprint
- alignment mode: currently `outer` for binary arithmetic materialization
- null/NaN policy: missing aligned operands materialize as column nulls/NaNs
- output ordering contract: exact indexes keep input order; unique outer alignment emits sorted labels; duplicate-aware outer alignment preserves left encounter order then right-only labels
- materialization reason: `series_binary_arithmetic_materialization`

Strict/hardened risk note:
- The witness is recorded before runtime admission decides allow/repair/reject, so rejected materialization attempts still leave an auditable semantic boundary.
- The witness does not change strict or hardened behavior. It records observed index and null/NaN contracts only.
- The stable fingerprint avoids persisting full index labels in the ledger while still giving deterministic equality evidence for replay.
