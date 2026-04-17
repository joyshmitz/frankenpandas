#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    assert!(
        fp_conformance::fuzz_scalar_cast_bytes(data).is_ok(),
        "scalar cast invariants should hold for all projected inputs"
    );
});
