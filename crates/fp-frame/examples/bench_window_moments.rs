//! Baseline/candidate benchmark for br-frankenpandas-g2veb.
//!
//! Times public rolling/expanding skew and kurt APIs on deterministic Float64
//! data, and emits a checksum over the visible outputs.

use std::time::Instant;

use fp_frame::Series;
use fp_index::IndexLabel;
use fp_types::Scalar;

const N: usize = 12_000;
const WINDOW: usize = 250;

fn next_value(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64 / (1_u64 << 53) as f64) * 1000.0 - 500.0
}

fn series() -> Series {
    let mut state = 0x6A09_E667_F3BC_C909_u64;
    let values: Vec<Scalar> = (0..N)
        .map(|idx| {
            let mut v = next_value(&mut state);
            if idx % 997 == 0 {
                v = 7.0;
            }
            Scalar::Float64(v)
        })
        .collect();
    let labels: Vec<IndexLabel> = (0..N as i64).map(IndexLabel::from).collect();
    Series::from_values("x", labels, values).unwrap()
}

fn checksum(series: &Series) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in series.values() {
        let bits = match value {
            Scalar::Float64(v) => v.to_bits(),
            Scalar::Null(_) => 0x7ff8_0000_0000_0000,
            other => other.to_string().len() as u64,
        };
        hash ^= bits;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

fn time_series(label: &str, f: impl FnOnce() -> Series) -> (Series, f64) {
    let started = Instant::now();
    let out = f();
    std::hint::black_box(&out);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    println!(
        "{label}: {elapsed_ms:.3} ms checksum {:016x}",
        checksum(&out)
    );
    (out, elapsed_ms)
}

fn main() {
    let s = series();
    println!("n={N} window={WINDOW}");

    let (rolling_skew, rolling_skew_ms) =
        time_series("rolling_skew", || s.rolling(WINDOW, None).skew().unwrap());
    let (rolling_kurt, rolling_kurt_ms) =
        time_series("rolling_kurt", || s.rolling(WINDOW, None).kurt().unwrap());
    let (expanding_skew, expanding_skew_ms) =
        time_series("expanding_skew", || s.expanding(Some(1)).skew().unwrap());
    let (expanding_kurt, expanding_kurt_ms) =
        time_series("expanding_kurt", || s.expanding(Some(1)).kurt().unwrap());

    let combined = checksum(&rolling_skew)
        ^ checksum(&rolling_kurt).rotate_left(13)
        ^ checksum(&expanding_skew).rotate_left(27)
        ^ checksum(&expanding_kurt).rotate_left(41);

    println!(
        "total_ms {:.3} combined_checksum {:016x}",
        rolling_skew_ms + rolling_kurt_ms + expanding_skew_ms + expanding_kurt_ms,
        combined
    );
}
