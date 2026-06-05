//! Bench + golden for dtype-aware elementwise: rint/neg/square/exp2/signbit/sign.
//!
//! Run: cargo run -p fp-columnar --example bench_elementwise2 --release
//!
//! Each scalar-looped self.values; an all-valid column now maps the op over its
//! contiguous Int64/Float64 buffer and re-ingests typed (dtype-preserving for
//! neg/square/sign; Bool for signbit; Float64 for rint/exp2). Bit-identical.

use fp_columnar::Column;
use fp_types::{DType, NullKind, Scalar};
use std::time::Instant;

fn fcol(v: Vec<f64>) -> Column {
    Column::from_f64_values(v)
}
fn icol(v: Vec<i64>) -> Column {
    Column::new(DType::Int64, v.into_iter().map(Scalar::Int64).collect()).unwrap()
}

fn dump(c: &Column) -> String {
    let mut s = format!("[{:?}]", c.dtype());
    for v in c.values() {
        match v {
            Scalar::Float64(f) if f.is_nan() => s.push_str("nan,"),
            Scalar::Float64(f) => s.push_str(&format!("f{},", f.to_bits())),
            Scalar::Int64(i) => s.push_str(&format!("i{i},")),
            Scalar::Bool(b) => s.push_str(&format!("b{b},")),
            Scalar::Null(_) => s.push_str("N,"),
            other => s.push_str(&format!("{other:?},")),
        }
    }
    s
}

fn golden() -> String {
    let f = fcol(vec![0.0, -0.0, 1.5, -1.5, 2.5, -2.5, 100.25, -7.0, 3.0]);
    let i = icol(vec![0, 1, -1, 5, -8, i64::MIN, 1_000_000]);
    let nf = Column::new(
        DType::Float64,
        vec![
            Scalar::Float64(-1.5),
            Scalar::Null(NullKind::NaN),
            Scalar::Float64(2.0),
        ],
    )
    .unwrap();
    let ni = Column::new(
        DType::Int64,
        vec![
            Scalar::Int64(-3),
            Scalar::Null(NullKind::NaN),
            Scalar::Int64(4),
        ],
    )
    .unwrap();

    let ops: Vec<(&str, fn(&Column) -> Result<Column, fp_columnar::ColumnError>)> = vec![
        ("rint", Column::rint),
        ("neg", Column::neg),
        ("square", Column::square),
        ("exp2", Column::exp2),
        ("signbit", Column::signbit),
        ("sign", Column::sign),
    ];

    let mut out = String::new();
    for (name, op) in &ops {
        out.push_str(&format!("{name}_f:{}\n", dump(&op(&f).unwrap())));
        out.push_str(&format!("{name}_i:{}\n", dump(&op(&i).unwrap())));
        out.push_str(&format!("{name}_nf:{}\n", dump(&op(&nf).unwrap())));
        out.push_str(&format!("{name}_ni:{}\n", dump(&op(&ni).unwrap())));
    }
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    let n: usize = 2_000_000;
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-6 - 1.0).collect();
    let col = fcol(data);

    let _ = col.sign().unwrap(); // warmup

    let t = Instant::now();
    let r = col.sign().unwrap();
    let d = t.elapsed();
    assert_eq!(r.len(), n);

    println!("TIMING n={n} sign={:.3}ms", d.as_secs_f64() * 1e3);
}
