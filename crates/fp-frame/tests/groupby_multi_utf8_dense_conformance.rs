//! No-mock conformance guard for the multi-Utf8-key dense groupby bypass
//! (multi_utf8_dense_grouping in the moments engine). The dense path fires only
//! for CONTIGUOUS Utf8 key columns; a Scalar-backed column (from_values) bails
//! as_utf8_contiguous and flows through the generic build_groups path. So the
//! same data built both ways must yield byte-identical results — a direct
//! dense-vs-generic bit-identity check — and both must match pandas 2.2.3.

use fp_columnar::Column;
use fp_frame::DataFrame;
use fp_index::{Index, IndexLabel};
use fp_types::Scalar;

// pandas groupby(['k1','k2']) on k1=[b,a,b,a,c] k2=[y,x,x,x,x] v=[1,2,3,4,5]:
//   order (a,x),(b,x),(b,y),(c,x); sum [6,3,1,5]; mean [3,3,1,5]; count [2,1,1,1]
const K1: [&str; 5] = ["b", "a", "b", "a", "c"];
const K2: [&str; 5] = ["y", "x", "x", "x", "x"];
const V: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

fn contig(keys: &[&str]) -> Column {
    let mut bytes = Vec::new();
    let mut offsets = vec![0usize];
    for s in keys {
        bytes.extend_from_slice(s.as_bytes());
        offsets.push(bytes.len());
    }
    Column::from_utf8_contiguous(bytes, offsets)
}
fn scalar(keys: &[&str]) -> Column {
    Column::from_values(keys.iter().map(|s| Scalar::Utf8((*s).to_string())).collect()).unwrap()
}

fn frame(k1: Column, k2: Column) -> DataFrame {
    let index = Index::new((0..5i64).map(IndexLabel::Int64).collect());
    let mut cols = std::collections::BTreeMap::new();
    cols.insert("k1".to_string(), k1);
    cols.insert("k2".to_string(), k2);
    cols.insert("v".to_string(), Column::from_values(V.iter().map(|x| Scalar::Float64(*x)).collect()).unwrap());
    DataFrame::new_with_column_order(index, cols, vec!["k1".into(), "k2".into(), "v".into()]).unwrap()
}

fn vals(df: &DataFrame) -> Vec<f64> {
    df.column("v")
        .unwrap()
        .values()
        .iter()
        .map(|s| match s {
            Scalar::Float64(x) => *x,
            Scalar::Int64(x) => *x as f64,
            other => panic!("unexpected {other:?}"),
        })
        .collect()
}
fn idx_labels(df: &DataFrame) -> Vec<String> {
    df.index().labels().iter().map(|l| l.to_string()).collect()
}

fn run(op: &str, df: &DataFrame) -> DataFrame {
    let gb = df.groupby(&["k1", "k2"]).unwrap();
    match op {
        "sum" => gb.sum().unwrap(),
        "mean" => gb.mean().unwrap(),
        "count" => gb.count().unwrap(),
        "min" => gb.min().unwrap(),
        "max" => gb.max().unwrap(),
        "median" => gb.median().unwrap(),
        _ => unreachable!(),
    }
}

#[test]
fn dense_matches_generic_and_pandas() {
    let expected: &[(&str, Vec<f64>)] = &[
        ("sum", vec![6.0, 3.0, 1.0, 5.0]),
        ("mean", vec![3.0, 3.0, 1.0, 5.0]),
        ("count", vec![2.0, 1.0, 1.0, 1.0]),
        ("min", vec![2.0, 3.0, 1.0, 5.0]),
        ("max", vec![4.0, 3.0, 1.0, 5.0]),
        ("median", vec![3.0, 3.0, 1.0, 5.0]),
    ];
    for (op, want) in expected {
        let dense = run(op, &frame(contig(&K1), contig(&K2))); // contiguous -> my dense path
        let generic = run(op, &frame(scalar(&K1), scalar(&K2))); // scalar-backed -> build_groups
        // dense == generic (bit-identity of the bypass)
        assert_eq!(vals(&dense), vals(&generic), "{op}: dense vs generic values");
        assert_eq!(idx_labels(&dense), idx_labels(&generic), "{op}: dense vs generic index");
        // both == pandas
        assert_eq!(&vals(&dense), want, "{op}: vs pandas values");
        assert_eq!(idx_labels(&dense), vec!["a, x", "b, x", "b, y", "c, x"], "{op}: sorted group order");
    }
}
