#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use fp_columnar::{ArithmeticOp, Column, ColumnError};
use fp_index::{Index, IndexError, IndexLabel, align_union, validate_alignment_plan};
use fp_runtime::{DecisionAction, EvidenceLedger, RuntimeMode, RuntimePolicy};
use fp_types::Scalar;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FrameError {
    #[error("index length ({index_len}) does not match column length ({column_len})")]
    LengthMismatch { index_len: usize, column_len: usize },
    #[error("duplicate index labels are unsupported in strict mode for MVP slice")]
    DuplicateIndexUnsupported,
    #[error("compatibility gate rejected operation: {0}")]
    CompatibilityRejected(String),
    #[error(transparent)]
    Column(#[from] ColumnError),
    #[error(transparent)]
    Index(#[from] IndexError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Series {
    name: String,
    index: Index,
    column: Column,
}

impl Series {
    pub fn new(name: impl Into<String>, index: Index, column: Column) -> Result<Self, FrameError> {
        if index.len() != column.len() {
            return Err(FrameError::LengthMismatch {
                index_len: index.len(),
                column_len: column.len(),
            });
        }

        Ok(Self {
            name: name.into(),
            index,
            column,
        })
    }

    pub fn from_values(
        name: impl Into<String>,
        index_labels: Vec<IndexLabel>,
        values: Vec<Scalar>,
    ) -> Result<Self, FrameError> {
        let index = Index::new(index_labels);
        let column = Column::from_values(values)?;
        Self::new(name, index, column)
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn index(&self) -> &Index {
        &self.index
    }

    #[must_use]
    pub fn column(&self) -> &Column {
        &self.column
    }

    #[must_use]
    pub fn values(&self) -> &[Scalar] {
        self.column.values()
    }

    pub fn add_with_policy(
        &self,
        other: &Self,
        policy: &RuntimePolicy,
        ledger: &mut EvidenceLedger,
    ) -> Result<Self, FrameError> {
        if self.index.has_duplicates() || other.index.has_duplicates() {
            policy.decide_unknown_feature(
                "index_alignment",
                "duplicate labels are not yet fully modeled",
                ledger,
            );
            if matches!(policy.mode, RuntimeMode::Strict) {
                return Err(FrameError::DuplicateIndexUnsupported);
            }
        }

        let plan = align_union(&self.index, &other.index);
        validate_alignment_plan(&plan)?;

        let left = self.column.reindex_by_positions(&plan.left_positions)?;
        let right = other.column.reindex_by_positions(&plan.right_positions)?;

        let action = policy.decide_join_admission(plan.union_index.len(), ledger);
        if matches!(action, DecisionAction::Reject) {
            return Err(FrameError::CompatibilityRejected(
                "runtime policy rejected alignment admission".to_owned(),
            ));
        }

        let column = left.binary_numeric(&right, ArithmeticOp::Add)?;

        let out_name = if self.name == other.name {
            self.name.clone()
        } else {
            format!("{}+{}", self.name, other.name)
        };

        Self::new(out_name, plan.union_index, column)
    }

    pub fn add(&self, other: &Self) -> Result<Self, FrameError> {
        let mut ledger = EvidenceLedger::new();
        self.add_with_policy(other, &RuntimePolicy::strict(), &mut ledger)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataFrame {
    index: Index,
    columns: BTreeMap<String, Column>,
}

impl DataFrame {
    pub fn new(index: Index, columns: BTreeMap<String, Column>) -> Result<Self, FrameError> {
        for column in columns.values() {
            if column.len() != index.len() {
                return Err(FrameError::LengthMismatch {
                    index_len: index.len(),
                    column_len: column.len(),
                });
            }
        }

        Ok(Self { index, columns })
    }

    pub fn from_series(series_list: Vec<Series>) -> Result<Self, FrameError> {
        if series_list.is_empty() {
            return Self::new(Index::new(Vec::new()), BTreeMap::new());
        }

        let mut series_iter = series_list.into_iter();
        let first = series_iter.next().expect("non-empty checked");
        let mut union_index = first.index.clone();

        let mut columns = BTreeMap::new();
        columns.insert(first.name, first.column);

        for series in series_iter {
            let plan = align_union(&union_index, &series.index);
            validate_alignment_plan(&plan)?;

            for column in columns.values_mut() {
                *column = column.reindex_by_positions(&plan.left_positions)?;
            }

            let aligned_column = series.column.reindex_by_positions(&plan.right_positions)?;
            columns.insert(series.name, aligned_column);
            union_index = plan.union_index;
        }

        Self::new(union_index, columns)
    }

    #[must_use]
    pub fn index(&self) -> &Index {
        &self.index
    }

    #[must_use]
    pub fn columns(&self) -> &BTreeMap<String, Column> {
        &self.columns
    }

    #[must_use]
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }
}

#[cfg(test)]
mod tests {
    use fp_runtime::{EvidenceLedger, RuntimePolicy};
    use fp_types::{NullKind, Scalar};

    use super::{DataFrame, FrameError, IndexLabel, Series};

    #[test]
    fn series_add_aligns_on_union_index() {
        let left = Series::from_values(
            "left",
            vec![1_i64.into(), 3_i64.into()],
            vec![Scalar::Int64(10), Scalar::Int64(30)],
        )
        .expect("left");
        let right = Series::from_values(
            "right",
            vec![2_i64.into(), 3_i64.into()],
            vec![Scalar::Int64(2), Scalar::Int64(4)],
        )
        .expect("right");

        let out = left
            .add_with_policy(
                &right,
                &RuntimePolicy::hardened(Some(100)),
                &mut EvidenceLedger::new(),
            )
            .expect("add should pass");

        assert_eq!(
            out.index().labels(),
            &[1_i64.into(), 3_i64.into(), 2_i64.into()]
        );
        assert_eq!(
            out.values(),
            &[
                Scalar::Null(NullKind::Null),
                Scalar::Int64(34),
                Scalar::Null(NullKind::Null)
            ]
        );
    }

    #[test]
    fn strict_mode_rejects_duplicate_indices() {
        let left = Series::from_values(
            "left",
            vec![IndexLabel::from("a"), IndexLabel::from("a")],
            vec![Scalar::Int64(1), Scalar::Int64(2)],
        )
        .expect("left");
        let right =
            Series::from_values("right", vec![IndexLabel::from("a")], vec![Scalar::Int64(3)])
                .expect("right");

        let err = left.add(&right).expect_err("strict mode should reject");
        assert!(matches!(err, FrameError::DuplicateIndexUnsupported));
    }

    #[test]
    fn dataframe_from_series_reindexes_existing_columns() {
        let s1 = Series::from_values(
            "a",
            vec![1_i64.into(), 2_i64.into()],
            vec![Scalar::Int64(1), Scalar::Int64(2)],
        )
        .expect("s1");
        let s2 = Series::from_values(
            "b",
            vec![2_i64.into(), 3_i64.into()],
            vec![Scalar::Int64(20), Scalar::Int64(30)],
        )
        .expect("s2");

        let df = DataFrame::from_series(vec![s1, s2]).expect("frame");
        assert_eq!(
            df.index().labels(),
            &[1_i64.into(), 2_i64.into(), 3_i64.into()]
        );
        assert_eq!(
            df.column("a").expect("a").values(),
            &[
                Scalar::Int64(1),
                Scalar::Int64(2),
                Scalar::Null(NullKind::Null)
            ]
        );
    }
}
