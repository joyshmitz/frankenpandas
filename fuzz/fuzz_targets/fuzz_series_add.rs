#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    assert!(
        fp_conformance::fuzz_series_add_bytes(data).is_ok(),
        "series add invariants should hold for all projected numeric series pairs"
    );
});
