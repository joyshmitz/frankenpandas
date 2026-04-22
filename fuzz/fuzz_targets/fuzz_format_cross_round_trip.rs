#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_CROSS_FORMAT_BYTES: usize = 512 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_CROSS_FORMAT_BYTES {
        return;
    }

    let _ = fp_conformance::fuzz_format_cross_round_trip_bytes(data);
});
