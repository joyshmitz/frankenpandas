#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_SEMANTIC_EQ_FUZZ_BYTES: usize = 4 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_SEMANTIC_EQ_FUZZ_BYTES {
        return;
    }

    let _ = fp_conformance::fuzz_semantic_eq_bytes(data);
});
