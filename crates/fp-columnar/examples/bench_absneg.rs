//! Column abs/neg over 5M i64. bench_absneg <n> <op>
use fp_columnar::Column;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(5_000_000);
    let op = a.get(2).map(String::as_str).unwrap_or("abs");
    let col = Column::from_i64_values((0..n as i64).map(|i| (i % 1000) - 500).collect());
    let mut best = u128::MAX;
    for _ in 0..6 {
        let t = std::time::Instant::now();
        let r = if op == "neg" { col.neg().unwrap() } else { col.abs().unwrap() };
        std::hint::black_box(r.len());
        best = best.min(t.elapsed().as_nanos());
    }
    println!("{op} i64 n={n}: best={best}ns ({:.2}ms)", best as f64 / 1e6);
}
