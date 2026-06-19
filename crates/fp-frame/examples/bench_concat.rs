//! Bench for the typed buffer concat levers (concat_series_columns + typed Int64 index
//! concat, tbrtu). Concatenates K Int64 series (each m rows) via
//! concat_series_with_ignore_index. Self-times best-of-N. Compare vs pandas pd.concat.
//!
//! Run: cargo run -p fp-frame --example bench_concat --release -- 1000000 8 30

use std::time::Instant;

use fp_columnar::Column;
use fp_frame::{concat_series_with_ignore_index, Series};
use fp_index::{Index, IndexLabel};
use fp_types::Scalar;

// Scalars-backed (from_values): as_i64_slice returns None for Int64 (no fast path).
fn build_one(m: usize, base: i64) -> Series {
    Series::from_values(
        "v",
        (0..m as i64).map(IndexLabel::Int64).collect(),
        (0..m as i64).map(|x| Scalar::Int64(x + base)).collect(),
    )
    .unwrap()
}

// Typed-backed (from_i64_values): as_i64_slice returns Some → typed concat path fires.
fn build_one_typed(m: usize, base: i64) -> Series {
    let index = Index::new_known_unique_int64_unit_range(0, m);
    let col = Column::from_i64_values((0..m as i64).map(|x| x + base).collect());
    Series::new("v", index, col).unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let total: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let m = total / k.max(1);

    let measure = |parts: &[Series]| -> u128 {
        let refs: Vec<&Series> = parts.iter().collect();
        let mut best = u128::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            let out = concat_series_with_ignore_index(&refs, true).expect("concat");
            let e = t.elapsed().as_nanos();
            std::hint::black_box(&out);
            if e < best {
                best = e;
            }
        }
        best
    };

    let scalars: Vec<Series> = (0..k).map(|i| build_one(m, (i * m) as i64)).collect();
    let typed: Vec<Series> = (0..k).map(|i| build_one_typed(m, (i * m) as i64)).collect();
    let best_scalars = measure(&scalars);
    let best_typed = measure(&typed);

    // Isolate: time ONLY building the result Series (lazy index + from_i64_values of a
    // pre-concatenated Vec<i64>), no concat iteration. Pinpoints whether the cost is in
    // result construction (Series::new/from_i64_values) or the concat extend/iteration.
    let flat: Vec<i64> = (0..total as i64).collect();
    let mut best_build = u128::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let idx = Index::new_known_unique_int64_unit_range(0, total);
        let col = Column::from_i64_values(flat.clone());
        let out = Series::new("v", idx, col).unwrap();
        let e = t.elapsed().as_nanos();
        std::hint::black_box(&out);
        if e < best_build {
            best_build = e;
        }
    }
    println!(
        "concat total={total} k={k} m={m} iters={iters}: scalars_backed={best_scalars}ns typed_backed={best_typed}ns direct_build={best_build}ns"
    );
}
