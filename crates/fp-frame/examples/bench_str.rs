//! String-op sweep: str.len, str.contains, str.startswith on 1M strings. fp uses
//! contiguous-Utf8 (apply_str_*); pandas str ops are object-dtype per-element. Head-to-head.
//!
//! Run: cargo run -p fp-frame --example bench_str --release -- 1000000 30

use std::time::Instant;

use fp_frame::Series;
use fp_index::IndexLabel;
use fp_types::Scalar;

fn best<F: FnMut()>(iters: usize, mut f: F) -> u128 {
    let mut b = u128::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        let e = t.elapsed().as_nanos();
        if e < b {
            b = e;
        }
    }
    b
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let labels: Vec<IndexLabel> = (0..n as i64).map(IndexLabel::Int64).collect();
    let vals: Vec<Scalar> = (0..n)
        .map(|i| Scalar::Utf8(format!("Row_{:08X}_Value_item", i)))
        .collect();
    let s = Series::from_values("v", labels, vals).unwrap();

    let len = best(iters, || {
        std::hint::black_box(s.str().len().expect("len"));
    });
    let contains = best(iters, || {
        std::hint::black_box(s.str().contains("Value").expect("contains"));
    });
    let startswith = best(iters, || {
        std::hint::black_box(s.str().startswith("Row_").expect("startswith"));
    });

    println!("str n={n}: len={len}ns contains={contains}ns startswith={startswith}ns");
}
