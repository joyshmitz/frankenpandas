#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_EVAL_FUZZ_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_EVAL_FUZZ_BYTES {
        return;
    }

    let _ = fp_conformance::fuzz_dataframe_eval_bytes(data);
});
