//! Bench + golden for Column::median / quantile — O(n) selection vs O(n log n).
//!
//! Run: cargo run -p fp-columnar --example bench_median_quantile --release
//!
//! nanmedian/nanquantile (fp-types) full-sorted the finite values; only the
//! middle order-statistic(s) are needed, so select_nth_unstable is O(n).
//! Bit-identical (order statistics depend only on values).

use fp_columnar::Column;
use fp_types::{DType, NullKind, Scalar};
use std::time::Instant;

fn fcol(v: Vec<f64>) -> Column {
    Column::from_f64_values(v)
}
fn icol(v: Vec<i64>) -> Column {
    Column::new(DType::Int64, v.into_iter().map(Scalar::Int64).collect()).unwrap()
}

fn ds(v: &Scalar) -> String {
    match v {
        Scalar::Float64(f) if f.is_nan() => "nan".into(),
        Scalar::Float64(f) => format!("f{}", f.to_bits()),
        Scalar::Int64(i) => format!("i{i}"),
        Scalar::Null(_) => "N".into(),
        other => format!("{other:?}"),
    }
}

fn golden() -> String {
    let mut out = String::new();
    // odd / even / dups / negatives / -0.0
    let odd = fcol(vec![3.0, 1.0, 2.0, 5.0, 4.0]);
    let even = fcol(vec![3.0, 1.0, 2.0, 5.0, 4.0, 6.0]);
    let dups = fcol(vec![2.0, 2.0, 2.0, 1.0, 3.0, 2.0]);
    let neg = fcol(vec![-5.0, 3.0, -1.0, 0.0, -0.0, 2.0]);
    out.push_str(&format!("median_odd:{}\n", ds(&odd.median())));
    out.push_str(&format!("median_even:{}\n", ds(&even.median())));
    out.push_str(&format!("median_dups:{}\n", ds(&dups.median())));
    out.push_str(&format!("median_neg:{}\n", ds(&neg.median())));
    out.push_str(&format!("median_one:{}\n", ds(&fcol(vec![7.5]).median())));
    out.push_str(&format!("median_empty:{}\n", ds(&fcol(vec![]).median())));
    out.push_str(&format!("median_int:{}\n", ds(&icol(vec![5, 1, 3, 2, 4]).median())));

    // quantiles across q
    for &q in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 0.333] {
        out.push_str(&format!("q{q}_odd:{}\n", ds(&odd.quantile(q))));
        out.push_str(&format!("q{q}_even:{}\n", ds(&even.quantile(q))));
        out.push_str(&format!("q{q}_dups:{}\n", ds(&dups.quantile(q))));
    }
    // quantile with nulls (scalar collect path), out-of-range, empty
    let nf = Column::new(
        DType::Float64,
        vec![Scalar::Float64(10.0), Scalar::Null(NullKind::NaN), Scalar::Float64(20.0), Scalar::Float64(30.0)],
    )
    .unwrap();
    out.push_str(&format!("q_null:{}\n", ds(&nf.quantile(0.5))));
    out.push_str(&format!("q_oor:{}\n", ds(&odd.quantile(1.5))));
    out.push_str(&format!("q_empty:{}\n", ds(&fcol(vec![]).quantile(0.5))));
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    let n: usize = 2_000_000;
    let mut x: u64 = 0x1234_5678_9abc_def0;
    let data: Vec<f64> = (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 11) as f64) / (1u64 << 53) as f64
        })
        .collect();
    let col = fcol(data);

    let _ = col.median(); // warmup

    let t = Instant::now();
    let mut sink = 0.0;
    for _ in 0..5 {
        if let Scalar::Float64(m) = col.median() {
            sink += m;
        }
        if let Scalar::Float64(m) = col.quantile(0.9) {
            sink += m;
        }
    }
    let d = t.elapsed();
    std::hint::black_box(sink);

    println!("TIMING n={n} median+q0.9_x5={:.3}ms", d.as_secs_f64() * 1e3);
}
