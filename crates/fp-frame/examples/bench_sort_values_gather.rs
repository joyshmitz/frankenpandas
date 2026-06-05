//! Bench + golden digest for Series::sort_values — typed gather lever.
//!
//! Run: cargo run -p fp-frame --example bench_sort_values_gather --release
//!
//! sort_values' typed radix argsort is O(n), so the permutation GATHER
//! (reorder_by_positions) dominates. It cloned a 32 B Scalar per row and
//! re-validated in Column::new; routing through Column::take_positions keeps
//! the gather on the contiguous typed buffer. Output is bit-identical.

use fp_frame::Series;
use fp_index::IndexLabel;
use fp_types::{NullKind, Scalar};
use std::time::Instant;

fn s_i64(vals: Vec<i64>) -> Series {
    let idx: Vec<IndexLabel> = (0..vals.len() as i64).map(IndexLabel::Int64).collect();
    Series::from_values("s", idx, vals.into_iter().map(Scalar::Int64).collect()).unwrap()
}

fn s_scalars(vals: Vec<Scalar>) -> Series {
    let idx: Vec<IndexLabel> = (0..vals.len() as i64).map(IndexLabel::Int64).collect();
    Series::from_values("s", idx, vals).unwrap()
}

fn golden() -> String {
    let mut out = String::new();
    // Int64 ascending/descending with ties (stable).
    let s = s_i64(vec![3, 1, 2, 1, 3]);
    out.push_str(&format!("i64_asc={:?}\n", s.sort_values(true).unwrap().values()));
    out.push_str(&format!("i64_desc={:?}\n", s.sort_values(false).unwrap().values()));

    // Float64 with NaN (NaN sorts last under na_position='last').
    let f = s_scalars(vec![
        Scalar::Float64(2.5),
        Scalar::Float64(f64::NAN),
        Scalar::Float64(-1.0),
        Scalar::Float64(2.5),
    ]);
    out.push_str(&format!("f64_asc={:?}\n", f.sort_values(true).unwrap().values()));

    // Nullable Int64 (Null mixed in) via na_position first/last.
    let ni = s_scalars(vec![
        Scalar::Int64(5),
        Scalar::Null(NullKind::NaN),
        Scalar::Int64(-2),
        Scalar::Int64(5),
    ]);
    out.push_str(&format!(
        "ni_last={:?}\n",
        ni.sort_values_na(true, "last").unwrap().values()
    ));
    out.push_str(&format!(
        "ni_first={:?}\n",
        ni.sort_values_na(true, "first").unwrap().values()
    ));

    // Utf8.
    let u = s_scalars(
        vec!["banana", "apple", "cherry", "apple"]
            .into_iter()
            .map(|x| Scalar::Utf8(x.to_string()))
            .collect(),
    );
    out.push_str(&format!("utf8_asc={:?}\n", u.sort_values(true).unwrap().values()));

    // Bool.
    let b = s_scalars(vec![
        Scalar::Bool(true),
        Scalar::Bool(false),
        Scalar::Bool(true),
    ]);
    out.push_str(&format!("bool_asc={:?}\n", b.sort_values(true).unwrap().values()));
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    let n: usize = 1_000_000;
    let mut x: u64 = 0x9e37_79b9;
    let vals: Vec<i64> = (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 16) as i64 % 1_000_000
        })
        .collect();
    let s = s_i64(vals);

    let _ = s.sort_values(true).unwrap(); // warmup

    let t = Instant::now();
    let r = s.sort_values(true).unwrap();
    let d = t.elapsed();
    assert_eq!(r.len(), n);

    println!("TIMING n={n} sort_values={:.3}ms", d.as_secs_f64() * 1e3);
}
