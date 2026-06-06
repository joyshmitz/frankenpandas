//! Bench + golden for Timedelta64 median/quantile — O(n) select_nth.
//!
//! Run: cargo run -p fp-columnar --example bench_td_median --release
//!
//! nanmedian/nanquantile's Timedelta arm full-sorted ns values; only the
//! middle order-statistic(s) are needed, so select_nth_unstable is O(n).
//! Bit-identical (order statistics depend only on values).

use fp_columnar::Column;
use fp_types::{DType, NullKind, Scalar};
use std::time::Instant;

fn tdcol(v: Vec<i64>) -> Column {
    Column::new(DType::Timedelta64, v.into_iter().map(Scalar::Timedelta64).collect()).unwrap()
}

fn ds(s: &Scalar) -> String {
    match s {
        Scalar::Timedelta64(ns) => format!("td{ns}"),
        Scalar::Float64(f) => format!("f{}", f.to_bits()),
        Scalar::Null(_) => "N".into(),
        o => format!("{o:?}"),
    }
}

fn golden() -> String {
    let mut out = String::new();
    // odd / even / dups
    let odd = tdcol(vec![300, 100, 200, 500, 400]);
    let even = tdcol(vec![300, 100, 200, 500, 400, 600]);
    let dups = tdcol(vec![200, 200, 200, 100, 300, 200]);
    out.push_str(&format!("median_odd:{}\n", ds(&odd.median())));
    out.push_str(&format!("median_even:{}\n", ds(&even.median())));
    out.push_str(&format!("median_dups:{}\n", ds(&dups.median())));
    out.push_str(&format!("median_one:{}\n", ds(&tdcol(vec![777]).median())));
    out.push_str(&format!("median_empty:{}\n", ds(&tdcol(vec![]).median())));

    for &q in &[0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 0.333] {
        out.push_str(&format!("q{q}_odd:{}\n", ds(&odd.quantile(q))));
        out.push_str(&format!("q{q}_even:{}\n", ds(&even.quantile(q))));
    }
    // with NaT (excluded)
    let nt = Column::new(
        DType::Timedelta64,
        vec![
            Scalar::Timedelta64(10),
            Scalar::Null(NullKind::NaT),
            Scalar::Timedelta64(30),
            Scalar::Timedelta64(20),
        ],
    )
    .unwrap();
    out.push_str(&format!("median_nat:{}\n", ds(&nt.median())));
    out.push_str(&format!("q0.5_nat:{}\n", ds(&nt.quantile(0.5))));
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    let n: usize = 2_000_000;
    let mut x: u64 = 0x7d4ed;
    let col = tdcol((0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 20) as i64 % 1_000_000_000
        })
        .collect());

    let _ = col.median(); // warmup

    let t = Instant::now();
    let mut sink = 0i64;
    for _ in 0..5 {
        if let Scalar::Timedelta64(m) = col.median() {
            sink = sink.wrapping_add(m);
        }
        if let Scalar::Timedelta64(m) = col.quantile(0.9) {
            sink = sink.wrapping_add(m);
        }
    }
    let d = t.elapsed();
    std::hint::black_box(sink);

    println!("TIMING n={n} td_median+q_x5={:.3}ms", d.as_secs_f64() * 1e3);
}
