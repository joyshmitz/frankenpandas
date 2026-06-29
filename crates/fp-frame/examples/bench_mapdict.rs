use fp_frame::Series;
use fp_columnar::Column;
use fp_index::Index;
use fp_types::Scalar;
fn main() {
    let g: Vec<String> = std::env::args().collect();
    let n: usize = g.get(1).and_then(|s| s.parse().ok()).unwrap_or(5_000_000);
    let col = Column::from_i64_values((0..n as i64).map(|i| i%1000).collect());
    let s = Series::new("s", Index::from_range(0, n as i64, 1), col).unwrap();
    let mapping: Vec<(Scalar,Scalar)> = (0..1000i64).map(|i| (Scalar::Int64(i), Scalar::Int64(i*2))).collect();
    let mut best=u128::MAX;
    for _ in 0..6 {
        let t=std::time::Instant::now();
        let r=s.map(&mapping).unwrap();
        std::hint::black_box(r.len());
        best=best.min(t.elapsed().as_nanos());
    }
    println!("map(dict) i64 5M: best={best}ns ({:.2}ms)", best as f64/1e6);
}
