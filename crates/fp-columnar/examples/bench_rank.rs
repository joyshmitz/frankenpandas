//! Bench + golden for Column::rank — O(n) counting-sort for bounded Int64.
//!
//! Run: cargo run -p fp-columnar --example bench_rank --release
//!
//! rank sorted (index,value) pairs O(n log n); an all-valid bounded-range Int64
//! column ranks via a value histogram + prefix sums (O(n)), bit-identical
//! across all 5 methods x asc/desc (Int64 compares exactly, no f64 gate).

use fp_columnar::Column;
use fp_types::{DType, NullKind, Scalar};
use std::time::Instant;

fn icol(v: Vec<i64>) -> Column {
    Column::new(DType::Int64, v.into_iter().map(Scalar::Int64).collect()).unwrap()
}

fn dump(c: &Column) -> String {
    let mut s = String::new();
    for v in c.values() {
        match v {
            Scalar::Float64(f) if f.is_nan() => s.push_str("nan,"),
            Scalar::Float64(f) => s.push_str(&format!("f{},", f.to_bits())),
            Scalar::Null(_) => s.push_str("N,"),
            o => s.push_str(&format!("{o:?},")),
        }
    }
    s
}

fn golden() -> String {
    let mut out = String::new();
    // ties at 3 and 1, negatives — exercises tie groups, dense, first-order.
    let a = icol(vec![3, 1, 3, -2, 1, 5, -2, 3]);
    for m in ["average", "min", "max", "first", "dense"] {
        out.push_str(&format!("{m}_asc:{}\n", dump(&a.rank(m, true).unwrap())));
        out.push_str(&format!("{m}_desc:{}\n", dump(&a.rank(m, false).unwrap())));
    }
    // all-same, single, empty
    out.push_str(&format!("same_avg:{}\n", dump(&icol(vec![5, 5, 5, 5]).rank("average", true).unwrap())));
    out.push_str(&format!("single:{}\n", dump(&icol(vec![9]).rank("min", true).unwrap())));
    out.push_str(&format!("empty:{}\n", dump(&icol(vec![]).rank("max", true).unwrap())));
    // wide-range -> sort fallback
    out.push_str(&format!("wide:{}\n", dump(&icol(vec![0, 1_000_000_000, 0, 5]).rank("min", true).unwrap())));
    // nullable -> sort fallback (missing stays NaN)
    let ni = Column::new(
        DType::Int64,
        vec![Scalar::Int64(2), Scalar::Null(NullKind::NaN), Scalar::Int64(1), Scalar::Int64(2)],
    )
    .unwrap();
    out.push_str(&format!("null_avg:{}\n", dump(&ni.rank("average", true).unwrap())));
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    let n: usize = 2_000_000;
    let mut x: u64 = 0x4a14;
    let col = icol((0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 42) as i64 % 50_000
        })
        .collect());

    let _ = col.rank("average", true).unwrap(); // warmup

    let t = Instant::now();
    let r = col.rank("average", true).unwrap();
    let d = t.elapsed();
    assert_eq!(r.len(), n);

    println!("TIMING n={n} rank_avg={:.3}ms", d.as_secs_f64() * 1e3);
}
