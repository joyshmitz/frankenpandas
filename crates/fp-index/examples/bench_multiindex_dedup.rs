//! Bench + golden digest for MultiIndex::duplicated/drop_duplicates
//! (br-frankenpandas-midedup).
//!
//! Run: cargo run -p fp-index --example bench_multiindex_dedup --release -- <n>
//!
//! The packed-key path hashes one identity-coded u64 per row instead of
//! allocating a Vec<IndexLabel> (Utf8 String clone) per row. `FP_NO_MIDEDUP=1`
//! forces the Vec<IndexLabel>-key baseline; the printed `chk` (FNV digest of the
//! duplicated mask + kept count) must match between the two runs.

use std::hint::black_box;
use std::time::Instant;

use fp_index::{DuplicateKeep, IndexLabel, MultiIndex};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    // ~50% duplicate tuples: low-cardinality 2 Utf8 levels.
    let l0: Vec<IndexLabel> = (0..n).map(|i| IndexLabel::Utf8(format!("g{}", i % 700))).collect();
    let l1: Vec<IndexLabel> = (0..n).map(|i| IndexLabel::Utf8(format!("h{}", (i / 7) % 700))).collect();
    let mi = MultiIndex::from_arrays(vec![l0, l1]).expect("mi");

    let mask = mi.duplicated(DuplicateKeep::First);
    let kept = mi.drop_duplicates().len();
    let mut chk: u64 = 0xcbf29ce484222325;
    for &b in &mask {
        chk = (chk ^ b as u64).wrapping_mul(0x100000001b3);
    }
    chk ^= (kept as u64).wrapping_mul(0x9E3779B97F4A7C15);

    let mut sink = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        sink ^= black_box(mi.duplicated(DuplicateKeep::First)).len();
    }
    let elapsed = start.elapsed();
    println!(
        "mi_dedup n={n} iters={iters}: {:.3} ms/iter (kept={kept} chk={chk:016x} sink={sink})",
        elapsed.as_secs_f64() * 1000.0 / iters as f64
    );
}
