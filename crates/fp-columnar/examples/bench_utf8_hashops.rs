//! Bench + golden for the Utf8 hash-family ops — SipHash -> FxHash completion.
//!
//! Run: cargo run -p fp-columnar --example bench_utf8_hashops --release -- [bench|golden]
//!
//! unique/factorize/isin/has_duplicates/mode/set-ops still keyed std-SipHash
//! HashSet/HashMap (the duplicated/value_counts sites were already Fx). Every
//! remaining site is order-independent: membership probes, first-seen output
//! order from a side Vec / the values iteration, or a full sort after
//! collection (mode) — so swapping the hasher is bit-identical.

use std::time::Instant;

use fp_columnar::Column;
use fp_types::{DType, NullKind, Scalar};

fn scol(v: Vec<String>) -> Column {
    Column::new(DType::Utf8, v.into_iter().map(Scalar::Utf8).collect()).unwrap()
}

fn dump(c: &Column) -> String {
    let mut s = String::new();
    for v in c.values() {
        match v {
            Scalar::Utf8(x) => s.push_str(&format!("{x}|")),
            Scalar::Int64(i) => s.push_str(&format!("{i}|")),
            Scalar::Bool(b) => s.push_str(&format!("{b}|")),
            Scalar::Float64(f) => s.push_str(&format!("f{}|", f.to_bits())),
            other => s.push_str(&format!("{other:?}|")),
        }
    }
    s
}

fn golden() -> String {
    let mut out = String::new();
    let a = scol(
        ["b", "a", "c", "a", "b", "d", "a", "", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );
    let with_na = Column::new(
        DType::Utf8,
        vec![
            Scalar::Utf8("x".into()),
            Scalar::Null(NullKind::Null),
            Scalar::Utf8("y".into()),
            Scalar::Utf8("x".into()),
            Scalar::Null(NullKind::Null),
        ],
    )
    .unwrap();
    out.push_str(&format!("unique:{}\n", dump(&a.unique().unwrap())));
    out.push_str(&format!("unique_na:{}\n", dump(&with_na.unique().unwrap())));
    let (codes, uniques) = a.factorize().unwrap();
    out.push_str(&format!("fact_codes:{}\n", dump(&codes)));
    out.push_str(&format!("fact_uniques:{}\n", dump(&uniques)));
    let (codes_na, uniques_na) = with_na.factorize().unwrap();
    out.push_str(&format!("factna_codes:{}\n", dump(&codes_na)));
    out.push_str(&format!("factna_uniques:{}\n", dump(&uniques_na)));
    let needles = vec![Scalar::Utf8("a".into()), Scalar::Utf8("d".into())];
    out.push_str(&format!("isin:{}\n", dump(&a.isin(&needles).unwrap())));
    out.push_str(&format!("hasdup:{}\n", a.has_duplicates()));
    out.push_str(&format!(
        "hasdup_uniq:{}\n",
        scol(vec!["p".into(), "q".into()]).has_duplicates()
    ));
    out.push_str(&format!("mode:{}\n", dump(&a.mode().unwrap())));
    // float mode exercises the FloatBits key incl. -0.0 normalization
    let f = Column::new(
        DType::Float64,
        vec![
            Scalar::Float64(1.5),
            Scalar::Float64(-0.0),
            Scalar::Float64(0.0),
            Scalar::Float64(1.5),
            Scalar::Float64(2.5),
            Scalar::Float64(0.0),
        ],
    )
    .unwrap();
    out.push_str(&format!("mode_f:{}\n", dump(&f.mode().unwrap())));
    let b = scol(["c", "e", "a", "e"].iter().map(|s| s.to_string()).collect());
    out.push_str(&format!("setdiff:{}\n", dump(&a.setdiff1d(&b).unwrap())));
    out.push_str(&format!(
        "intersect:{}\n",
        dump(&a.intersect1d(&b).unwrap())
    ));
    out.push_str(&format!("setxor:{}\n", dump(&a.setxor1d(&b).unwrap())));
    out.push_str(&format!("in1d:{}\n", dump(&a.in1d(&b).unwrap())));
    out.push_str(&format!(
        "delete:{}\n",
        dump(&a.delete(&[0, 3, 8]).unwrap())
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

    let n: usize = 1_000_000;
    let mut rng = Rng(0xFEED_FACE_CAFE_0042);
    let col = scol(
        (0..n)
            .map(|_| {
                let k = rng.next() % 200_000;
                format!("user_{k:06}")
            })
            .collect(),
    );
    let needles: Vec<Scalar> = (0..1000)
        .map(|i| Scalar::Utf8(format!("user_{:06}", i * 137)))
        .collect();

    let _ = col.unique().unwrap(); // warmup

    let iters = 10;
    let mut b_unique = f64::INFINITY;
    let mut b_fact = f64::INFINITY;
    let mut b_isin = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let u = col.unique().unwrap();
        b_unique = b_unique.min(t.elapsed().as_secs_f64());
        std::hint::black_box(&u);

        let t = Instant::now();
        let f = col.factorize().unwrap();
        b_fact = b_fact.min(t.elapsed().as_secs_f64());
        std::hint::black_box(&f);

        let t = Instant::now();
        let i = col.isin(&needles).unwrap();
        b_isin = b_isin.min(t.elapsed().as_secs_f64());
        std::hint::black_box(&i);
    }
    println!(
        "utf8_hashops n={n} unique={:.3}ms factorize={:.3}ms isin={:.3}ms",
        b_unique * 1e3,
        b_fact * 1e3,
        b_isin * 1e3
    );
}
