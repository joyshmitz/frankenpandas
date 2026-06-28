//! Column::dropna over 5M nullable (1/4 gaps). bench_dropna <n> <dt>
use fp_columnar::{Column, ValidityMask};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(5_000_000);
    let dt = a.get(2).map(String::as_str).unwrap_or("f64");
    let mut validity = ValidityMask::all_valid(n);
    for i in (0..n).step_by(4) { validity.set(i, false); }
    let col = if dt == "i64" {
        Column::from_i64_values_with_validity((0..n as i64).collect(), validity)
    } else {
        Column::from_f64_values_with_validity((0..n).map(|i| i as f64).collect(), validity)
    };
    let mut best = u128::MAX;
    for _ in 0..6 {
        let t = std::time::Instant::now();
        let r = col.dropna().unwrap();
        std::hint::black_box(r.len());
        let e = t.elapsed().as_nanos();
        if e < best { best = e; }
    }
    println!("dropna {dt} n={n}: best={best}ns ({:.2}ms)", best as f64 / 1e6);
}
