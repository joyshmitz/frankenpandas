#![forbid(unsafe_code)]

use std::collections::HashMap;

use fp_columnar::{Column, ColumnError};
use fp_frame::{FrameError, Series};
use fp_index::{Index, IndexLabel};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JoinedSeries {
    pub index: Index,
    pub left_values: Column,
    pub right_values: Column,
}

#[derive(Debug, Error)]
pub enum JoinError {
    #[error(transparent)]
    Frame(#[from] FrameError),
    #[error(transparent)]
    Column(#[from] ColumnError),
}

pub fn join_series(
    left: &Series,
    right: &Series,
    join_type: JoinType,
) -> Result<JoinedSeries, JoinError> {
    // AG-02: borrowed-key HashMap eliminates right-index label clones during build phase.
    let mut right_map = HashMap::<&IndexLabel, Vec<usize>>::new();
    for (pos, label) in right.index().labels().iter().enumerate() {
        right_map.entry(label).or_default().push(pos);
    }

    let mut out_labels = Vec::new();
    let mut left_positions = Vec::<Option<usize>>::new();
    let mut right_positions = Vec::<Option<usize>>::new();

    for (left_pos, label) in left.index().labels().iter().enumerate() {
        if let Some(matches) = right_map.get(label) {
            for right_pos in matches {
                out_labels.push(label.clone());
                left_positions.push(Some(left_pos));
                right_positions.push(Some(*right_pos));
            }
            continue;
        }

        if matches!(join_type, JoinType::Left) {
            out_labels.push(label.clone());
            left_positions.push(Some(left_pos));
            right_positions.push(None);
        }
    }

    let left_values = left.column().reindex_by_positions(&left_positions)?;
    let right_values = right.column().reindex_by_positions(&right_positions)?;

    Ok(JoinedSeries {
        index: Index::new(out_labels),
        left_values,
        right_values,
    })
}

#[cfg(test)]
mod tests {
    use fp_types::{NullKind, Scalar};

    use super::{JoinType, join_series};
    use fp_frame::Series;

    #[test]
    fn inner_join_multiplies_cardinality_for_duplicates() {
        let left = Series::from_values(
            "left",
            vec!["k".into(), "k".into(), "x".into()],
            vec![Scalar::Int64(1), Scalar::Int64(2), Scalar::Int64(3)],
        )
        .expect("left");

        let right = Series::from_values(
            "right",
            vec!["k".into(), "k".into()],
            vec![Scalar::Int64(10), Scalar::Int64(20)],
        )
        .expect("right");

        let out = join_series(&left, &right, JoinType::Inner).expect("join");
        assert_eq!(out.index.labels().len(), 4);
        assert_eq!(
            out.left_values.values(),
            &[
                Scalar::Int64(1),
                Scalar::Int64(1),
                Scalar::Int64(2),
                Scalar::Int64(2)
            ]
        );
    }

    #[test]
    fn left_join_injects_missing_for_unmatched_right_rows() {
        let left = Series::from_values(
            "left",
            vec!["a".into(), "b".into()],
            vec![Scalar::Int64(1), Scalar::Int64(2)],
        )
        .expect("left");
        let right =
            Series::from_values("right", vec!["a".into()], vec![Scalar::Int64(10)]).expect("right");

        let out = join_series(&left, &right, JoinType::Left).expect("join");
        assert_eq!(
            out.right_values.values(),
            &[Scalar::Int64(10), Scalar::Null(NullKind::Null)]
        );
    }
}
