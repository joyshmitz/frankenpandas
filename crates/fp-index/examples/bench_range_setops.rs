//! Head-to-head-friendly RangeIndex set-op microbench.
//!
//! Run:
//!   cargo run -p fp-index --example bench_range_setops --release -- 1000000 200 overlap

use std::{hint::black_box, time::Instant};

use fp_index::RangeIndex;

fn best_ns(iters: usize, mut f: impl FnMut() -> usize) -> (u128, usize) {
    let mut sink = 0usize;
    for _ in 0..3 {
        sink ^= black_box(f());
    }
    let mut best = u128::MAX;
    for _ in 0..iters {
        let started = Instant::now();
        sink ^= black_box(f());
        let elapsed = started.elapsed().as_nanos();
        best = best.min(elapsed);
    }
    (best, sink)
}

fn ranges(n: usize, scenario: &str) -> (RangeIndex, RangeIndex) {
    let n = i64::try_from(n).expect("n fits i64");
    match scenario {
        "adjacent" => (
            RangeIndex::new(0, n, 1).expect("valid left range"),
            RangeIndex::new(n, n * 2, 1).expect("valid right range"),
        ),
        "descending" => (
            RangeIndex::new(n, 0, -1).expect("valid left range"),
            RangeIndex::new(n / 2, -(n / 2), -1).expect("valid right range"),
        ),
        _ => (
            RangeIndex::new(0, n, 1).expect("valid left range"),
            RangeIndex::new(n / 2, n + n / 2, 1).expect("valid right range"),
        ),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000_000);
    let iters: usize = args
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);
    let scenario = args.get(3).map_or("overlap", String::as_str);
    let (left, right) = ranges(n, scenario);

    let (intersection_ns, intersection_sink) = best_ns(iters, || left.intersection(&right).len());
    let (union_ns, union_sink) = best_ns(iters, || left.union(&right).len());
    let (difference_ns, difference_sink) = best_ns(iters, || left.difference(&right).len());
    let (symmetric_difference_ns, symmetric_difference_sink) =
        best_ns(iters, || left.symmetric_difference(&right).len());
    let sink = intersection_sink ^ union_sink ^ difference_sink ^ symmetric_difference_sink;

    println!(
        "range_setops n={n} scenario={scenario} intersection_ns={intersection_ns} union_ns={union_ns} difference_ns={difference_ns} symmetric_difference_ns={symmetric_difference_ns} sink={sink}"
    );
}
