# Dependency Upgrade Log

Date of first entry: 2026-04-21

## Updates

### asupersync: 0.2.0 → 0.3.0 — 2026-04-21
- **Scope:** `crates/fp-runtime/Cargo.toml` (optional dep, `default-features = false`, gated by the `fp-runtime/asupersync` feature). `Cargo.lock` refreshed.
- **Landed in:** upstream commit `9e8b574`, which beat cc's parallel bump to origin; cc's identical 0.2 → 0.3 edit was dropped during rebase.
- **Breaking review:** The only source-breaking change on the 0.2.x → 0.3.0 path was v0.2.9 widening `ObjectParams.source_blocks` from `u8` to `u16` — not touched by fp-runtime. The v0.3.0 delta is overwhelmingly a coordinated dependency refresh (digest-0.11 wave, hashbrown 0.17, rusqlite 0.39, lz4_flex 0.13, signal-hook 0.4, rayon 1.12) plus three concurrency bug fixes (parking_lot self-deadlock in observability, DNS-coalesce waiter count, TLS close_notify fail-closed). Our only public-API touchpoint is `asupersync::Outcome<T, E>` (variants `Ok`/`Err`/`Cancelled`/`Panicked`), still exported from the crate root in v0.3.0 with the same shape.
- **Migration:** none. fp-runtime call sites in `src/lib.rs` (`outcome_to_action`) compile unchanged.
- **Tests:** `rch exec -- cargo test -p fp-runtime --features asupersync` → 30 passed, 0 failed. `rch exec -- cargo clippy -p fp-runtime --features asupersync --all-targets -- -D warnings` → clean. `rch exec -- cargo check -p fp-runtime --features asupersync` → OK.
