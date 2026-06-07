//! Bench + golden for Column::rank on Float64 — O(n) stable radix permutation.
//!
//! Run: cargo run -p fp-columnar --example bench_rank_f64 --release -- [bench|golden]
//!
//! The Float64 rank path comparison-sorted (index,&Scalar) pairs O(n log n). An
//! all-valid, NaN-free Float64 column now ranks via the stable LSD radix perm
//! (the same one sort_values/argsort use), then walks tie groups with f64 `==`.
//! Bit-identical: f64_radix_key normalizes -0.0->0.0 (matching partial_cmp's
//! -0.0==0.0) and the radix is stable (ties keep original order). NaN routes to
//! the unchanged comparator fallback.

use std::time::Instant;

use fp_columnar::Column;
use fp_types::{DType, Scalar};

fn fcol(v: Vec<f64>) -> Column {
    Column::from_f64_values(v)
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
    // ties at 3.0 and 1.0, negatives, a -0.0 vs 0.0 mix — exercises tie groups,
    // dense, first-order, and the -0.0 normalization.
    let a = fcol(vec![3.0, 1.0, 3.0, -2.5, 1.0, 5.0, -2.5, 3.0, 0.0, -0.0]);
    for m in ["average", "min", "max", "first", "dense"] {
        out.push_str(&format!("{m}_asc:{}\n", dump(&a.rank(m, true).unwrap())));
        out.push_str(&format!("{m}_desc:{}\n", dump(&a.rank(m, false).unwrap())));
    }
    out.push_str(&format!(
        "same:{}\n",
        dump(&fcol(vec![5.0, 5.0, 5.0]).rank("average", true).unwrap())
    ));
    out.push_str(&format!(
        "single:{}\n",
        dump(&fcol(vec![9.0]).rank("min", true).unwrap())
    ));
    out.push_str(&format!(
        "empty:{}\n",
        dump(&fcol(vec![]).rank("max", true).unwrap())
    ));
    // NaN -> comparator fallback (NaN is missing, ranked NaN).
    let nan_col = Column::new(
        DType::Float64,
        vec![
            Scalar::Float64(2.0),
            Scalar::Float64(f64::NAN),
            Scalar::Float64(1.0),
            Scalar::Float64(2.0),
        ],
    )
    .unwrap();
    out.push_str(&format!(
        "nan_avg:{}\n",
        dump(&nan_col.rank("average", true).unwrap())
    ));
    out
}

/// splitmix64 — deterministic, no rand dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn main() {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bench".to_string());

    if mode == "golden" {
        print!("{}", golden());
        return;
    }

    let n: usize = 2_000_000;
    let mut rng = Rng(0x51A4_2C7E);
    // spread of floats with ties (mod keeps the distinct count bounded)
    let col = fcol(
        (0..n)
            .map(|_| ((rng.next() % 100_000) as f64) * 0.25)
            .collect(),
    );

    let _ = col.rank("average", true).unwrap(); // warmup

    let iters = 20;
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let r = col.rank("average", true).unwrap();
        let d = t.elapsed().as_secs_f64();
        std::hint::black_box(&r);
        if d < best {
            best = d;
        }
    }
    println!("rank_f64 n={n} avg best={:.3}ms", best * 1e3);
}
