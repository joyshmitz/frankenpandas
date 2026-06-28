//! Column::searchsorted_values over 5M sorted i64 with 1M i64 needles. bench_searchsorted <n> <m>
use fp_columnar::Column;
use fp_types::Scalar;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(5_000_000);
    let m: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let col = Column::from_i64_values((0..n as i64).map(|i| i * 3).collect());
    let needles: Vec<Scalar> = (0..m as i64).map(|i| Scalar::Int64(i * 7)).collect();
    let mut best = u128::MAX;
    for _ in 0..6 {
        let t = std::time::Instant::now();
        let r = col.searchsorted_values(&needles, "left").unwrap();
        std::hint::black_box(r.len());
        let e = t.elapsed().as_nanos();
        if e < best {
            best = e;
        }
    }
    println!("searchsorted {n}/{m}: best={best}ns ({:.2}ms)", best as f64 / 1e6);
}
