#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_EXCEL_BYTES: usize = 512 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_EXCEL_BYTES {
        return;
    }

    let _ = fp_conformance::fuzz_excel_io_bytes(data);
});
