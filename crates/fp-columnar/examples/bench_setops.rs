//! Bench + golden for np set-ops setdiff1d/intersect1d/setxor1d/in1d.
//!
//! Run: cargo run -p fp-columnar --example bench_setops --release
//!
//! Each scanned the other operand linearly with `.any(semantic_eq)` per value
//! — O(N·M) (setdiff1d also O(N²) for its first-seen dedup). A hash-set built
//! once makes them O(N+M). Output order/dedup preserved exactly.

use std::time::Instant;

use fp_columnar::Column;
use fp_types::{DType, NullKind, Scalar};

fn icol(v: Vec<i64>) -> Column {
    Column::new(DType::Int64, v.into_iter().map(Scalar::Int64).collect()).unwrap()
}
fn fcol(v: Vec<f64>) -> Column {
    Column::from_f64_values(v)
}
fn scol(v: &[&str]) -> Column {
    Column::new(
        DType::Utf8,
        v.iter().map(|s| Scalar::Utf8((*s).to_string())).collect(),
    )
    .unwrap()
}

fn dump(c: &Column) -> String {
    let mut s = String::new();
    for v in c.values() {
        match v {
            Scalar::Int64(i) => s.push_str(&format!("{i},")),
            Scalar::Float64(f) if f.is_nan() => s.push_str("nan,"),
            Scalar::Float64(f) => s.push_str(&format!("f{},", f.to_bits())),
            Scalar::Utf8(x) => s.push_str(&format!("{x},")),
            Scalar::Bool(b) => s.push_str(&format!("{b},")),
            Scalar::Null(_) => s.push_str("N,"),
            o => s.push_str(&format!("{o:?},")),
        }
    }
    s
}

fn golden() -> String {
    let mut out = String::new();
    // dups in self (first-seen order matters for setdiff1d), overlap, with -0.0.
    let a = icol(vec![5, 1, 3, 1, 9, 5, 7]);
    let b = icol(vec![3, 9, 9, 2]);
    out.push_str(&format!("sd_ab:{}\n", dump(&a.setdiff1d(&b).unwrap())));
    out.push_str(&format!("int_ab:{}\n", dump(&a.intersect1d(&b).unwrap())));
    out.push_str(&format!("xor_ab:{}\n", dump(&a.setxor1d(&b).unwrap())));
    out.push_str(&format!("in_ab:{}\n", dump(&a.in1d(&b).unwrap())));

    // floats incl -0.0/0.0 equivalence
    let fa = fcol(vec![1.5, -0.0, 2.5, 3.0]);
    let fb = fcol(vec![0.0, 3.0]);
    out.push_str(&format!("sd_f:{}\n", dump(&fa.setdiff1d(&fb).unwrap())));
    out.push_str(&format!("in_f:{}\n", dump(&fa.in1d(&fb).unwrap())));

    // strings
    let sa = scol(&["apple", "banana", "apple", "cherry"]);
    let sb = scol(&["banana", "date"]);
    out.push_str(&format!("sd_s:{}\n", dump(&sa.setdiff1d(&sb).unwrap())));
    out.push_str(&format!("xor_s:{}\n", dump(&sa.setxor1d(&sb).unwrap())));

    // with missing
    let na = Column::new(
        DType::Int64,
        vec![
            Scalar::Int64(1),
            Scalar::Null(NullKind::NaN),
            Scalar::Int64(2),
        ],
    )
    .unwrap();
    out.push_str(&format!("in_na:{}\n", dump(&na.in1d(&b).unwrap())));
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    // N values, M-element other with moderate overlap.
    let n: usize = 200_000;
    let m: usize = 200_000;
    let a = icol((0..n as i64).map(|i| i % (n as i64 / 2)).collect());
    let b = icol((0..m as i64).map(|i| i % (m as i64 / 2) + 50_000).collect());

    let _ = a.in1d(&b).unwrap(); // warmup

    let t = Instant::now();
    let r = a.in1d(&b).unwrap();
    let d = t.elapsed();
    assert_eq!(r.len(), n);

    println!("TIMING n={n} m={m} in1d={:.3}ms", d.as_secs_f64() * 1e3);
}
