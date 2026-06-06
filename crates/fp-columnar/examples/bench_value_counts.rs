//! Bench + golden for Column::value_counts — O(N) hashmap tally.
//!
//! Run: cargo run -p fp-columnar --example bench_value_counts --release
//!
//! value_counts did a linear `counts.iter().find(semantic_eq)` per value
//! (O(N·distinct)); a key-hashmap makes it O(N). First-seen order preserved so
//! the stable count-sort breaks ties identically. Bit-identical.

use fp_columnar::Column;
use fp_types::{DType, NullKind, Scalar};
use std::time::Instant;

fn icol(v: Vec<i64>) -> Column {
    Column::new(DType::Int64, v.into_iter().map(Scalar::Int64).collect()).unwrap()
}
fn fcol(v: Vec<f64>) -> Column {
    Column::from_f64_values(v)
}

fn dump(pair: &(Column, Column)) -> String {
    let (vals, counts) = pair;
    let mut s = String::new();
    for (v, c) in vals.values().iter().zip(counts.values()) {
        let vs = match v {
            Scalar::Int64(i) => format!("{i}"),
            Scalar::Float64(f) if f.is_nan() => "nan".into(),
            Scalar::Float64(f) => format!("f{}", f.to_bits()),
            Scalar::Null(_) => "N".into(),
            o => format!("{o:?}"),
        };
        let cs = match c {
            Scalar::Int64(i) => format!("{i}"),
            Scalar::Float64(f) => format!("f{}", f.to_bits()),
            o => format!("{o:?}"),
        };
        s.push_str(&format!("{vs}={cs},"));
    }
    s
}

fn golden() -> String {
    let mut out = String::new();
    // ties + first-seen order: 3 appears first among the count-2 group.
    let a = icol(vec![3, 1, 3, 2, 1, 4, 2, 3]); // 3:3, 1:2, 2:2, 4:1
    out.push_str(&format!("desc:{}\n", dump(&a.value_counts_with_options(false, true, false, true).unwrap())));
    out.push_str(&format!("asc:{}\n", dump(&a.value_counts_with_options(false, true, true, true).unwrap())));
    out.push_str(&format!("nosort:{}\n", dump(&a.value_counts_with_options(false, false, false, true).unwrap())));
    out.push_str(&format!("norm:{}\n", dump(&a.value_counts_with_options(true, true, false, true).unwrap())));

    // with missing, dropna false (NaN bucket appended before sort)
    let nf = fcol(vec![1.0, f64::NAN, 1.0, 2.0, f64::NAN, f64::NAN, -0.0, 0.0]);
    out.push_str(&format!("nan_keep:{}\n", dump(&nf.value_counts_with_options(false, true, false, false).unwrap())));
    out.push_str(&format!("nan_drop:{}\n", dump(&nf.value_counts_with_options(false, true, false, true).unwrap())));
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    // All-distinct values (distinct == N) => the old linear find is O(N^2).
    let n: usize = 60_000;
    let col = icol((0..n as i64).collect());

    let _ = col.value_counts_with_options(false, true, false, true).unwrap(); // warmup

    let t = Instant::now();
    let r = col.value_counts_with_options(false, true, false, true).unwrap();
    let d = t.elapsed();
    std::hint::black_box(&r);

    println!("TIMING n={n} value_counts={:.3}ms", d.as_secs_f64() * 1e3);
}
