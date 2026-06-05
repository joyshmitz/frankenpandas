//! Isolates `Column::take_positions` on an all-valid Int64 column — the gather
//! behind DataFrame filter/iloc/sort/reindex/take. Before this lever the Int64
//! all-valid path rebuilt a `Vec<Scalar::Int64>` (32 B/elem) while Float64 kept
//! a typed lazy buffer; this measures the gather of a realistic AoS-built Int64
//! column under a 50%-selectivity boolean filter.
//!
//! Modes:
//!   take_bench golden <n>        -> deterministic output digest (sha proof)
//!   take_bench <n> <iters>       -> timed loop (hyperfine target)

use std::time::Instant;

use fp_columnar::Column;
use fp_types::{DType, Scalar};

fn build_column(n: usize) -> Column {
    // Built from Scalars (the common columnar-from-data shape), which caches a
    // typed Int64 buffer so as_i64_slice can serve the gather.
    let values: Vec<Scalar> = (0..n as i64)
        .map(|i| Scalar::Int64(i.wrapping_mul(2_654_435_761) ^ (i >> 3)))
        .collect();
    Column::new(DType::Int64, values).expect("int64 column")
}

fn positions(n: usize) -> Vec<usize> {
    // Keep every even index — a 50%-selectivity filter result.
    (0..n).filter(|i| i % 2 == 0).collect()
}

fn digest(col: &Column) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(col.len() as u64);
    for v in col.values() {
        match v {
            Scalar::Int64(i) => mix(*i as u64),
            Scalar::Null(_) => mix(0xDEAD_BEEF),
            other => mix(format!("{other:?}")
                .bytes()
                .fold(0u64, |a, b| a.wrapping_mul(131).wrapping_add(u64::from(b)))),
        }
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("golden") {
        let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5_000);
        let col = build_column(n);
        let pos = positions(n);
        let out = col.take_positions(&pos);
        println!(
            "take_golden n={n} out_len={} digest={:016x}",
            out.len(),
            digest(&out)
        );
        return;
    }

    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let col = build_column(n);
    let pos = positions(n);

    let start = Instant::now();
    let mut sink: usize = 0;
    for _ in 0..iters {
        let out = col.take_positions(&pos);
        sink = sink.wrapping_add(out.len());
    }
    let elapsed = start.elapsed();
    eprintln!(
        "take_bench n={n} iters={iters} {:.3}s ({:.3} ms/iter), sink={sink}",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / iters as f64,
    );
}
