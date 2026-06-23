//! Bench + golden digest for MultiIndex::duplicated/drop_duplicates/nunique
//! (br-frankenpandas-midedup).
//!
//! Run: cargo run -p fp-index --example bench_multiindex_dedup --release -- <n>
//!
//! The packed-key path hashes one identity-coded u64 per row instead of
//! allocating a Vec<IndexLabel> (Utf8 String clone) per row. `FP_NO_MIDEDUP=1`
//! forces the Vec<IndexLabel>-key baseline; the printed `chk` (FNV digest of the
//! duplicated mask + kept count) must match between the two runs.

use std::{hint::black_box, time::Instant};

use fp_index::{DuplicateKeep, IndexLabel, MultiIndex};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    // ~50% duplicate tuples: low-cardinality 2 Utf8 levels.
    let l0: Vec<IndexLabel> = (0..n)
        .map(|i| IndexLabel::Utf8(format!("g{}", i % 700)))
        .collect();
    let l1: Vec<IndexLabel> = (0..n)
        .map(|i| IndexLabel::Utf8(format!("h{}", (i / 7) % 700)))
        .collect();
    let mi = MultiIndex::from_arrays(vec![l0, l1]).expect("mi");

    let mask = mi.duplicated(DuplicateKeep::First);
    let kept = mi.drop_duplicates().len();
    let nunique = mi.nunique();
    let unique_len = mi.unique().len();
    if nunique != unique_len {
        eprintln!("nunique mismatch: nunique={nunique} unique_len={unique_len}");
        std::process::exit(2);
    }
    let mut chk: u64 = 0xcbf29ce484222325;
    for &b in &mask {
        chk = (chk ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
    chk ^= u64::try_from(kept)
        .unwrap_or(u64::MAX)
        .wrapping_mul(0x9E3779B97F4A7C15);
    chk ^= u64::try_from(nunique)
        .unwrap_or(u64::MAX)
        .wrapping_mul(0xBF58476D1CE4E5B9);

    let mut sink = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        sink ^= black_box(mi.duplicated(DuplicateKeep::First)).len();
    }
    let duplicated_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        sink ^= black_box(mi.nunique());
    }
    let nunique_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        sink ^= black_box(mi.unique().len());
    }
    let unique_len_elapsed = start.elapsed();

    println!(
        "mi_dedup n={n} iters={iters}: duplicated_ms={:.3} nunique_ms={:.3} unique_len_ms={:.3} kept={kept} nunique={nunique} chk={chk:016x} sink={sink}",
        duplicated_elapsed.as_secs_f64() * 1000.0 / iters as f64,
        nunique_elapsed.as_secs_f64() * 1000.0 / iters as f64,
        unique_len_elapsed.as_secs_f64() * 1000.0 / iters as f64
    );
}
