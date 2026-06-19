//! Bench for the typed buffer concat levers (concat_series_columns + typed Int64 index
//! concat, tbrtu). Concatenates K Int64 series (each m rows) via
//! concat_series_with_ignore_index. Self-times best-of-N. Compare vs pandas pd.concat.
//!
//! Run: cargo run -p fp-frame --example bench_concat --release -- 1000000 8 30

use std::time::Instant;

use fp_frame::{concat_series_with_ignore_index, Series};
use fp_index::IndexLabel;
use fp_types::Scalar;

fn build_one(m: usize, base: i64) -> Series {
    Series::from_values(
        "v",
        (0..m as i64).map(IndexLabel::Int64).collect(),
        (0..m as i64).map(|x| Scalar::Int64(x + base)).collect(),
    )
    .unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let total: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let m = total / k.max(1);
    let parts: Vec<Series> = (0..k).map(|i| build_one(m, (i * m) as i64)).collect();
    let refs: Vec<&Series> = parts.iter().collect();

    let mut best = u128::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let out = concat_series_with_ignore_index(&refs, true).expect("concat");
        let e = t.elapsed().as_nanos();
        std::hint::black_box(&out);
        if e < best {
            best = e;
        }
    }
    println!("concat total={total} k={k} m={m} iters={iters}: best={best}ns");
}
