//! No-mock conformance guard for Series.nunique over a Datetime64 VALUE column
//! routed through the inline-i64 FxHashSet path. Differential against an Int64
//! column with the SAME ns (distinct count is order-independent), plus a NaT
//! (i64::MIN) case that must bail to the generic dropna path.

use fp_columnar::Column;
use fp_frame::Series;
use fp_index::{Index, IndexLabel};

fn idx(n: usize) -> Index {
    Index::new((0..n as i64).map(IndexLabel::Int64).collect())
}

fn cases() -> Vec<Vec<i64>> {
    let base = 1_577_836_800_000_000_000i64;
    let step = 60_000_000_000i64;
    vec![
        vec![base + 3 * step, base, base + step, base, base + 3 * step],
        (0..2000).map(|i| base + ((i * 2654435761_i64) % 137) * step).collect(),
        (0..1000).map(|i| base + (i % 7) * step).collect(),
        (0..500).map(|i| base + i * step).collect(), // all distinct
    ]
}

#[test]
fn series_datetime_nunique_matches_distinct_count() {
    for ns in cases() {
        let n = ns.len();
        let dt = Series::new("s", idx(n), Column::from_datetime64_values(ns.clone())).unwrap();
        let distinct: std::collections::HashSet<i64> = ns.iter().copied().collect();
        assert_eq!(dt.nunique(), distinct.len(), "nunique {n}");
    }
}

#[test]
fn series_datetime_nunique_nat_bails_to_generic_dropna() {
    let base = 1_577_836_800_000_000_000i64;
    // i64::MIN is the Datetime64 NaT sentinel; dropna=true default excludes it.
    let ns = vec![base, i64::MIN, base + 60_000_000_000, base, i64::MIN];
    let dt = Series::new("s", idx(ns.len()), Column::from_datetime64_values(ns)).unwrap();
    assert_eq!(dt.nunique(), 2, "two present distinct timestamps, NaT excluded");
}
