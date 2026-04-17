#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    assert!(fp_conformance::fuzz_column_arith_bytes(data).is_ok());
});
