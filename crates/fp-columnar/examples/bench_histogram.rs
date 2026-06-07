//! Bench + golden for Column::histogram — O(N·log B) binary-search bin assign.
//!
//! Run: cargo run -p fp-columnar --example bench_histogram --release
//!
//! histogram linear-scanned all bins per value (O(N·B)); strictly-increasing
//! edges admit a partition_point binary search (O(N·log B)). Bit-identical:
//! right-open bins with inclusive final edge; out-of-range dropped; non-strict
//! edges keep the linear path.

use std::time::Instant;

use fp_columnar::Column;
use fp_types::{DType, NullKind, Scalar};

fn fcol(v: Vec<f64>) -> Column {
    Column::from_f64_values(v)
}

fn dump(c: &Column) -> String {
    let mut s = String::new();
    for v in c.values() {
        if let Scalar::Int64(i) = v {
            s.push_str(&format!("{i},"));
        }
    }
    s
}

fn golden() -> String {
    let mut out = String::new();
    // Values incl boundaries, below-range, above-range, NaN/Inf, exact last edge.
    let data = fcol(vec![
        -1.0,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.0,
        2.5,
        3.0,
        5.0, // 3.0 = last edge, 5.0 above
        f64::NAN,
        f64::INFINITY,
    ]);
    // strict edges [0,1,2,3]: bins [0,1),[1,2),[2,3]
    out.push_str(&format!(
        "strict:{}\n",
        dump(&data.histogram(&[0.0, 1.0, 2.0, 3.0]).unwrap())
    ));
    // single bin
    out.push_str(&format!(
        "one_bin:{}\n",
        dump(&data.histogram(&[0.0, 3.0]).unwrap())
    ));
    // many bins
    let edges: Vec<f64> = (0..=10).map(|i| i as f64 * 0.5).collect();
    out.push_str(&format!(
        "many:{}\n",
        dump(&data.histogram(&edges).unwrap())
    ));
    // non-strict edges (duplicate) -> linear fallback path
    out.push_str(&format!(
        "nonstrict:{}\n",
        dump(&data.histogram(&[0.0, 1.0, 1.0, 2.0, 3.0]).unwrap())
    ));
    // nullable column
    let nf = Column::new(
        DType::Float64,
        vec![
            Scalar::Float64(0.5),
            Scalar::Null(NullKind::NaN),
            Scalar::Float64(2.5),
        ],
    )
    .unwrap();
    out.push_str(&format!(
        "null:{}\n",
        dump(&nf.histogram(&[0.0, 1.0, 2.0, 3.0]).unwrap())
    ));
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    let n: usize = 2_000_000;
    let n_bins = 1000usize;
    let mut x: u64 = 0x000c_0ffe_e123;
    let data: Vec<f64> = (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64) / (1u64 << 53) as f64 * (n_bins as f64)
        })
        .collect();
    let col = fcol(data);
    let edges: Vec<f64> = (0..=n_bins).map(|i| i as f64).collect();

    let _ = col.histogram(&edges).unwrap(); // warmup

    let t = Instant::now();
    let r = col.histogram(&edges).unwrap();
    let d = t.elapsed();
    assert_eq!(r.len(), n_bins);

    println!(
        "TIMING n={n} bins={n_bins} histogram={:.3}ms",
        d.as_secs_f64() * 1e3
    );
}
