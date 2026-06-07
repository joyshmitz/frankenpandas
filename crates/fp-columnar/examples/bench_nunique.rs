//! Bench + golden for Column::nunique — dense direct-address for bounded Int64.
//!
//! Run: cargo run -p fp-columnar --example bench_nunique --release
//!
//! nunique counted distinct via fp-types nannunique (SipHash); an all-valid
//! bounded-range Int64 column counts via a seen-bitset indexed by (v-min),
//! hash-free. Non-bounded / non-Int64 / nullable keep the nannunique path.

use std::time::Instant;

use fp_columnar::Column;
use fp_types::{DType, NullKind, Scalar};

fn icol(v: Vec<i64>) -> Column {
    Column::new(DType::Int64, v.into_iter().map(Scalar::Int64).collect()).unwrap()
}

fn ds(s: &Scalar) -> String {
    match s {
        Scalar::Int64(i) => format!("{i}"),
        o => format!("{o:?}"),
    }
}

fn golden() -> String {
    let mut out = String::new();
    // bounded int (dense)
    out.push_str(&format!(
        "dense:{}\n",
        ds(&icol(vec![3, 1, 3, -2, 1, 5, -2, 3]).nunique())
    ));
    out.push_str(&format!("single:{}\n", ds(&icol(vec![7, 7, 7]).nunique())));
    out.push_str(&format!("empty:{}\n", ds(&icol(vec![]).nunique())));
    // wide range -> nannunique fallback
    out.push_str(&format!(
        "wide:{}\n",
        ds(&icol(vec![0, 1_000_000_000, 0, 5]).nunique())
    ));
    // nullable: dropna true vs false (nannunique path; missing skipped/+1)
    let ni = Column::new(
        DType::Int64,
        vec![
            Scalar::Int64(2),
            Scalar::Null(NullKind::NaN),
            Scalar::Int64(2),
            Scalar::Int64(9),
        ],
    )
    .unwrap();
    out.push_str(&format!(
        "null_drop:{}\n",
        ds(&ni.nunique_with_dropna(true))
    ));
    out.push_str(&format!(
        "null_keep:{}\n",
        ds(&ni.nunique_with_dropna(false))
    ));
    // utf8
    let s = Column::new(
        DType::Utf8,
        ["b", "a", "b", "c", "a"]
            .iter()
            .map(|x| Scalar::Utf8((*x).into()))
            .collect(),
    )
    .unwrap();
    out.push_str(&format!("utf8:{}\n", ds(&s.nunique())));
    out
}

fn main() {
    let g = golden();
    print!("GOLDEN_BEGIN\n{g}GOLDEN_END\n");

    let n: usize = 4_000_000;
    let mut x: u64 = 0x000f_ee15;
    let col = icol(
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 40) as i64 % 200_000
            })
            .collect(),
    );

    let _ = col.nunique(); // warmup

    let t = Instant::now();
    let mut sink = 0i64;
    for _ in 0..3 {
        if let Scalar::Int64(c) = col.nunique() {
            sink = sink.wrapping_add(c);
        }
    }
    let d = t.elapsed();
    std::hint::black_box(sink);

    println!("TIMING n={n} nunique_x3={:.3}ms", d.as_secs_f64() * 1e3);
}
