//! Series.factorize() on a single Float64 or wide-Int64 column, measurable vs
//! pandas `pd.factorize(s)`. Run: cargo run -p fp-frame --example factorize_bench --release -- 1000000 100000 30 f64

use std::time::Instant;

use fp_columnar::Column;
use fp_frame::Series;
use fp_index::Index;

fn best<F: FnMut()>(iters: usize, mut f: F) -> u128 {
    for _ in 0..3 {
        f();
    }
    let mut b = u128::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_nanos());
    }
    b
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let distinct: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let dtype = args.get(4).map(String::as_str).unwrap_or("f64");

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let index = Index::new_known_unique_int64_unit_range(0, rows);
    let s = if dtype == "i64" {
        let data: Vec<i64> = (0..rows)
            .map(|_| (next() % distinct) as i64 * 0x1_0000_0001_i64)
            .collect();
        Series::new("c".to_string(), index, Column::from_i64_values(data)).expect("series")
    } else {
        let data: Vec<f64> = (0..rows).map(|_| (next() % distinct) as f64 * 1.5).collect();
        Series::new("c".to_string(), index, Column::from_f64_values(data)).expect("series")
    };

    let ns = best(iters, || {
        std::hint::black_box(s.factorize().expect("factorize"));
    });
    let (_codes, uniq) = s.factorize().unwrap();
    eprintln!(
        "factorize_{dtype}: rows={rows} distinct={distinct} uniq={} best={:.1}us",
        uniq.len(),
        ns as f64 / 1000.0
    );
    println!("{:.1}", ns as f64 / 1000.0);
}
