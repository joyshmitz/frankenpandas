//! Bench + golden for binary elementwise ufuncs (atan2/hypot/copysign/fmod/
//! float_power/maximum/minimum/fmax/fmin) — typed two-buffer fast path.
//!
//! Run: cargo run -p fp-columnar --example bench_binary_ufuncs --release
//!
//! Each scalar-looped both self.values and other.values; when both columns are
//! all-valid numeric, typed_float_binary maps f over the two contiguous buffers
//! and re-ingests via from_f64_values. Bit-identical (incl NaN-producing cases:
//! from_f64_values re-marks NaN missing exactly as Self::new does).

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
            Scalar::Null(_) => s.push_str("N,"),
            other => s.push_str(&format!("{other:?},")),
        }
    }
    s
}

fn golden() -> String {
    // a/b chosen to exercise NaN-producing (fmod by 0, float_power neg^0.5),
    // Inf, -0.0, negatives, ties.
    let a = fcol(vec![1.0, -1.0, 2.0, -0.0, 5.0, -4.0, 0.0, 3.0]);
    let b = fcol(vec![2.0, 0.5, 0.0, 1.0, -5.0, 0.5, 0.0, -3.0]);
    let ai = icol(vec![3, -7, 10, 0, 5]);
    let bi = icol(vec![2, 4, -3, 0, 5]);
    let na = Column::new(
        DType::Float64,
        vec![Scalar::Float64(1.0), Scalar::Null(NullKind::NaN), Scalar::Float64(2.0)],
    )
    .unwrap();
    let nb = Column::new(
        DType::Float64,
        vec![Scalar::Float64(2.0), Scalar::Float64(3.0), Scalar::Null(NullKind::NaN)],
    )
    .unwrap();

    let ops: Vec<(&str, fn(&Column, &Column) -> Result<Column, fp_columnar::ColumnError>)> = vec![
        ("atan2", Column::atan2),
        ("hypot", Column::hypot),
        ("copysign", Column::copysign),
        ("fmod", Column::fmod),
        ("float_power", Column::float_power),
        ("maximum", Column::maximum),
        ("minimum", Column::minimum),
        ("fmax", Column::fmax),
        ("fmin", Column::fmin),
    ];

    let mut out = String::new();
    for (name, op) in &ops {
        out.push_str(&format!("{name}_ff:{}\n", dump(&op(&a, &b).unwrap())));
        out.push_str(&format!("{name}_ii:{}\n", dump(&op(&ai, &bi).unwrap())));
        out.push_str(&format!("{name}_nn:{}\n", dump(&op(&na, &nb).unwrap())));
    }
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    let n: usize = 2_000_000;
    let x = fcol((0..n).map(|i| (i as f64) * 1e-6 - 1.0).collect());
    let y = fcol((0..n).map(|i| 1.0 + (i as f64) * 2e-6).collect());

    // Headline a dispatch-bound op (maximum): the per-element work is trivial,
    // so the 32 B Scalar materialization/clone on both sides is the cost.
    // (Transcendental ops like atan2 are compute-bound and gain less.)
    let _ = x.maximum(&y).unwrap(); // warmup

    let t = Instant::now();
    let r = x.maximum(&y).unwrap();
    let d = t.elapsed();
    assert_eq!(r.len(), n);

    println!("TIMING n={n} maximum={:.3}ms", d.as_secs_f64() * 1e3);
}
