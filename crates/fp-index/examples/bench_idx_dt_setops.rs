//! Index union/intersection/isin over Datetime64 indexes @200k (time-series
//! alignment). Run: bench_idx_dt_setops <n> <op>
use fp_index::{Index, IndexLabel};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let op = a.get(2).map(String::as_str).unwrap_or("union");
    let base = 1_577_836_800_000_000_000i64; // 2020-01-01 ns
    let step = 60_000_000_000i64; // 1 minute
    // self: base + i*step ; other: shifted by n/2 (half overlap)
    let a_ns: Vec<i64> = (0..n).map(|i| base + i as i64 * step).collect();
    let b_ns: Vec<i64> = (0..n).map(|i| base + (i + n / 2) as i64 * step).collect();
    let idx = Index::from_datetime64(a_ns);
    let other = Index::from_datetime64(b_ns.clone());
    let needles: Vec<IndexLabel> = b_ns.iter().map(|&v| IndexLabel::Datetime64(v)).collect();
    let mut best = u128::MAX;
    for _ in 0..6 {
        let t = std::time::Instant::now();
        match op {
            "union" => {
                std::hint::black_box(idx.union(&other));
            }
            "intersection" => {
                std::hint::black_box(idx.intersection(&other));
            }
            "isin" => {
                std::hint::black_box(idx.isin(&needles));
            }
            "difference" => {
                std::hint::black_box(idx.difference(&other));
            }
            _ => panic!("op"),
        }
        let e = t.elapsed().as_nanos();
        if e < best {
            best = e;
        }
    }
    println!("idx_dt_{op} n={n}: best={best}ns");
}
