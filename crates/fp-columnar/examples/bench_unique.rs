//! Bench + golden for Column::unique — dense direct-address for bounded Int64.
//!
//! Run: cargo run -p fp-columnar --example bench_unique --release
//!
//! unique used a std SipHash HashSet; an all-valid bounded-range Int64 column
//! dedups via a seen-bitset indexed by (v - min), hash-free, first-seen order
//! preserved. Non-bounded / non-Int64 / nullable columns keep the HashSet path.

use fp_columnar::Column;
use fp_types::{DType, NullKind, Scalar};
use std::time::Instant;

fn icol(v: Vec<i64>) -> Column {
    Column::new(DType::Int64, v.into_iter().map(Scalar::Int64).collect()).unwrap()
}

fn dump(c: &Column) -> String {
    let mut s = format!("[{:?}]", c.dtype());
    for v in c.values() {
        match v {
            Scalar::Int64(i) => s.push_str(&format!("{i},")),
            Scalar::Float64(f) => s.push_str(&format!("f{},", f.to_bits())),
            Scalar::Utf8(x) => s.push_str(&format!("{x},")),
            Scalar::Null(_) => s.push_str("N,"),
            o => s.push_str(&format!("{o:?},")),
        }
    }
    s
}

fn golden() -> String {
    let mut out = String::new();
    // bounded int with dups + negatives (dense path), first-seen order.
    out.push_str(&format!("dense:{}\n", dump(&icol(vec![3, 1, 3, -2, 1, 5, -2, 3]).unique().unwrap())));
    out.push_str(&format!("single:{}\n", dump(&icol(vec![7, 7, 7]).unique().unwrap())));
    out.push_str(&format!("empty:{}\n", dump(&icol(vec![]).unique().unwrap())));
    // huge-range int (range > cap) -> HashSet fallback
    out.push_str(&format!("wide:{}\n", dump(&icol(vec![0, 1_000_000_000, 0, 5]).unique().unwrap())));
    // nullable int -> HashSet path (missing skipped)
    let ni = Column::new(
        DType::Int64,
        vec![Scalar::Int64(2), Scalar::Null(NullKind::NaN), Scalar::Int64(2), Scalar::Int64(9)],
    )
    .unwrap();
    out.push_str(&format!("nullable:{}\n", dump(&ni.unique().unwrap())));
    // strings -> HashSet path
    let s = Column::new(
        DType::Utf8,
        ["b", "a", "b", "c", "a"].iter().map(|x| Scalar::Utf8((*x).into())).collect(),
    )
    .unwrap();
    out.push_str(&format!("utf8:{}\n", dump(&s.unique().unwrap())));
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    // Bounded Int64, low cardinality but large n (heavy dedup work).
    let n: usize = 2_000_000;
    let mut x: u64 = 0xbead_5;
    let col = icol((0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 40) as i64 % 100_000
        })
        .collect());

    let _ = col.unique().unwrap(); // warmup

    let t = Instant::now();
    let r = col.unique().unwrap();
    let d = t.elapsed();
    std::hint::black_box(&r);

    println!("TIMING n={n} unique={:.3}ms", d.as_secs_f64() * 1e3);
}
