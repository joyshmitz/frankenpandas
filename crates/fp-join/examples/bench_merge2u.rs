//! merge on TWO Scalar-backed Utf8 keys (from_values). bench_merge2u <n> <card> <how>
use std::collections::BTreeMap;
use fp_columnar::Column;
use fp_frame::DataFrame;
use fp_index::{Index, IndexLabel};
use fp_join::{JoinType, merge_dataframes_on};
use fp_types::Scalar;
use std::hint::black_box;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(2_000_000);
    let card: i64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let how = a.get(3).map(String::as_str).unwrap_or("inner");
    let key = |i: usize, salt: i64| ((i as i64).wrapping_mul(2654435761).wrapping_add(salt) >> 13) % card;
    let scol = |rows: usize, f: &dyn Fn(usize) -> String| {
        Column::from_values((0..rows).map(|i| Scalar::Utf8(f(i))).collect()).unwrap()
    };
    let mut lm = BTreeMap::new();
    lm.insert("k1".to_string(), scol(n, &|i| format!("a{:05}", key(i, 0))));
    lm.insert("k2".to_string(), scol(n, &|i| format!("b{:05}", key(i, 7))));
    lm.insert("lv".to_string(), Column::from_f64_values((0..n).map(|i| i as f64).collect()));
    let left = DataFrame::new_with_column_order(Index::new((0..n as i64).map(IndexLabel::Int64).collect()), lm, vec!["k1".into(),"k2".into(),"lv".into()]).unwrap();
    let m = (card * card) as usize;
    let mut rm = BTreeMap::new();
    rm.insert("k1".to_string(), scol(m, &|i| format!("a{:05}", (i as i64) / card)));
    rm.insert("k2".to_string(), scol(m, &|i| format!("b{:05}", (i as i64) % card)));
    rm.insert("rv".to_string(), Column::from_f64_values((0..m).map(|i| i as f64).collect()));
    let right = DataFrame::new_with_column_order(Index::new((0..m as i64).map(IndexLabel::Int64).collect()), rm, vec!["k1".into(),"k2".into(),"rv".into()]).unwrap();
    let jt = match how { "left"=>JoinType::Left, "outer"=>JoinType::Outer, "right"=>JoinType::Right, _=>JoinType::Inner };
    let on = ["k1","k2"];
    let mut best = u128::MAX;
    for _ in 0..6 {
        let t = Instant::now();
        black_box(merge_dataframes_on(&left, &right, &on, jt).unwrap());
        best = best.min(t.elapsed().as_nanos());
    }
    println!("merge2u_{how} n={n} card={card}: {:.2}ms", best as f64/1e6);
}
