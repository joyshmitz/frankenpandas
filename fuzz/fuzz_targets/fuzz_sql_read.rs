#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_SQL_FUZZ_BYTES: usize = 4 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_SQL_FUZZ_BYTES {
        return;
    }

    let _ = fp_conformance::fuzz_read_sql_bytes(data);
});
