#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_FIXTURE_BYTES: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FIXTURE_BYTES {
        return;
    }

    let _ = fp_conformance::fuzz_fixture_parse_bytes(data);
});
