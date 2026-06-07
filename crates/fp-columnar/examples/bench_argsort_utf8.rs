//! Bench + golden for Column::argsort_with / sort_values on Utf8 — MSD byte radix.
//!
//! Run: cargo run -p fp-columnar --example bench_argsort_utf8 --release -- [bench|golden]
//!
//! The Utf8 sort path comparison-sorts (index, &Scalar) pairs O(n log n) with a
//! Scalar enum dispatch per comparison. An all-valid Utf8 column can instead use
//! a STABLE MSD byte-radix sort (counting passes over bytes, comparison-free at
//! scale). Bit-identical: `String::cmp` is byte-lexicographic with
//! shorter-prefix-first, which is exactly MSD order with an end-of-string bucket
//! that sorts before every byte (ascending) / after every byte (descending);
//! counting passes are stable so equal strings keep original order, matching the
//! stable `sort_by`.

use std::time::Instant;

use fp_columnar::Column;
use fp_types::{DType, Scalar};

fn scol(v: Vec<&str>) -> Column {
    Column::new(
        DType::Utf8,
        v.into_iter().map(|s| Scalar::Utf8(s.to_string())).collect(),
    )
    .unwrap()
}

fn dump_perm(p: &[usize]) -> String {
    p.iter().map(|i| format!("{i},")).collect()
}

fn dump_col(c: &Column) -> String {
    let mut s = String::new();
    for v in c.values() {
        match v {
            Scalar::Utf8(x) => s.push_str(&format!("{x}|")),
            other => s.push_str(&format!("{other:?}|")),
        }
    }
    s
}

fn golden() -> String {
    let mut out = String::new();
    // prefixes, ties (dup strings), empty string, multi-byte UTF-8, lengths
    // straddling the prefix boundary — exercises EOS bucket order + stability.
    let a = scol(vec![
        "banana",
        "apple",
        "app",
        "",
        "apple",
        "cherry",
        "äpfel",
        "app",
        "apricot",
        "b",
        "applesauce",
        "Zebra",
        "zebra",
        "apple\u{0}x",
        "apple",
    ]);
    out.push_str(&format!("asc:{}\n", dump_perm(&a.argsort_with(true))));
    out.push_str(&format!("desc:{}\n", dump_perm(&a.argsort_with(false))));
    out.push_str(&format!(
        "sv_asc:{}\n",
        dump_col(&a.sort_values(true).unwrap())
    ));
    out.push_str(&format!(
        "sv_desc:{}\n",
        dump_col(&a.sort_values(false).unwrap())
    ));
    // all-equal: pure stability check
    let e = scol(vec!["x", "x", "x", "x"]);
    out.push_str(&format!("same_asc:{}\n", dump_perm(&e.argsort_with(true))));
    out.push_str(&format!(
        "same_desc:{}\n",
        dump_perm(&e.argsort_with(false))
    ));
    // single + empty
    out.push_str(&format!(
        "single:{}\n",
        dump_perm(&scol(vec!["q"]).argsort_with(true))
    ));
    out.push_str(&format!(
        "empty:{}\n",
        dump_perm(&scol(vec![]).argsort_with(true))
    ));
    // missing values -> fallback path (na last either direction)
    let m = Column::new(
        DType::Utf8,
        vec![
            Scalar::Utf8("m".into()),
            Scalar::Null(fp_types::NullKind::Null),
            Scalar::Utf8("a".into()),
            Scalar::Utf8("m".into()),
        ],
    )
    .unwrap();
    out.push_str(&format!("na_asc:{}\n", dump_perm(&m.argsort_with(true))));
    out.push_str(&format!("na_desc:{}\n", dump_perm(&m.argsort_with(false))));
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

    let n: usize = 1_000_000;
    let mut rng = Rng(0xC0FF_EE00_5EED_1234);
    // ticker-like keys with shared prefixes and ties (~200k distinct).
    let owned: Vec<String> = (0..n)
        .map(|_| {
            let k = rng.next() % 200_000;
            let grp = k % 7;
            match grp {
                0 => format!("sym_{k}"),
                1 => format!("symbol_long_{k}"),
                2 => format!("id{k}"),
                _ => format!("user_{:06}", k),
            }
        })
        .collect();
    let col = Column::new(
        DType::Utf8,
        owned.iter().map(|s| Scalar::Utf8(s.clone())).collect(),
    )
    .unwrap();

    let _ = col.argsort_with(true); // warmup

    let iters = 10;
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let p = col.argsort_with(true);
        let d = t.elapsed().as_secs_f64();
        std::hint::black_box(&p);
        if d < best {
            best = d;
        }
    }
    println!("argsort_utf8 n={n} best={:.3}ms", best * 1e3);
}
