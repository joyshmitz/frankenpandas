#![forbid(unsafe_code)]

use fp_types::{DType, NullKind, Scalar, TypeError, cast_scalar_owned, common_dtype, infer_dtype};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidityMask {
    bits: Vec<bool>,
}

impl ValidityMask {
    #[must_use]
    pub fn from_values(values: &[Scalar]) -> Self {
        let bits = values.iter().map(|value| !value.is_missing()).collect();
        Self { bits }
    }

    #[must_use]
    pub fn bits(&self) -> &[bool] {
        &self.bits
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Column {
    dtype: DType,
    values: Vec<Scalar>,
    validity: ValidityMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ColumnError {
    #[error("column length mismatch: left={left}, right={right}")]
    LengthMismatch { left: usize, right: usize },
    #[error(transparent)]
    Type(#[from] TypeError),
}

impl Column {
    /// Construct a column, coercing values to the target dtype.
    /// AG-03: takes ownership of the values vec and uses `cast_scalar_owned`
    /// to skip cloning when values already have the correct dtype.
    pub fn new(dtype: DType, values: Vec<Scalar>) -> Result<Self, ColumnError> {
        let needs_coercion = values.iter().any(|v| {
            let d = v.dtype();
            d != dtype && d != DType::Null
        });

        let coerced = if needs_coercion {
            values
                .into_iter()
                .map(|value| cast_scalar_owned(value, dtype))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // No coercion needed: values already match dtype.
            // Only remap Null variants to the dtype-specific missing marker.
            values
                .into_iter()
                .map(|value| match value {
                    Scalar::Null(_) => Scalar::missing_for_dtype(dtype),
                    other => other,
                })
                .collect()
        };

        let validity = ValidityMask::from_values(&coerced);

        Ok(Self {
            dtype,
            values: coerced,
            validity,
        })
    }

    pub fn from_values(values: Vec<Scalar>) -> Result<Self, ColumnError> {
        let dtype = infer_dtype(&values)?;
        Self::new(dtype, values)
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[must_use]
    pub fn values(&self) -> &[Scalar] {
        &self.values
    }

    #[must_use]
    pub fn value(&self, idx: usize) -> Option<&Scalar> {
        self.values.get(idx)
    }

    #[must_use]
    pub fn validity(&self) -> &ValidityMask {
        &self.validity
    }

    pub fn reindex_by_positions(&self, positions: &[Option<usize>]) -> Result<Self, ColumnError> {
        let values = positions
            .iter()
            .map(|slot| match slot {
                Some(idx) => self
                    .values
                    .get(*idx)
                    .cloned()
                    .unwrap_or_else(|| Scalar::missing_for_dtype(self.dtype)),
                None => Scalar::missing_for_dtype(self.dtype),
            })
            .collect::<Vec<_>>();

        Self::new(self.dtype, values)
    }

    pub fn binary_numeric(&self, right: &Self, op: ArithmeticOp) -> Result<Self, ColumnError> {
        if self.len() != right.len() {
            return Err(ColumnError::LengthMismatch {
                left: self.len(),
                right: right.len(),
            });
        }

        let mut out_dtype = common_dtype(self.dtype, right.dtype)?;
        if matches!(out_dtype, DType::Bool) {
            out_dtype = DType::Int64;
        }
        if matches!(op, ArithmeticOp::Div) {
            out_dtype = DType::Float64;
        }

        let values = self
            .values
            .iter()
            .zip(&right.values)
            .map(|(left, right)| {
                if left.is_missing() || right.is_missing() {
                    return Ok::<_, ColumnError>(if left.is_nan() || right.is_nan() {
                        Scalar::Null(NullKind::NaN)
                    } else {
                        Scalar::missing_for_dtype(out_dtype)
                    });
                }

                let lhs = left.to_f64()?;
                let rhs = right.to_f64()?;
                let result = match op {
                    ArithmeticOp::Add => lhs + rhs,
                    ArithmeticOp::Sub => lhs - rhs,
                    ArithmeticOp::Mul => lhs * rhs,
                    ArithmeticOp::Div => lhs / rhs,
                };

                if matches!(out_dtype, DType::Int64)
                    && result.is_finite()
                    && result == result.trunc()
                    && result >= i64::MIN as f64
                    && result <= i64::MAX as f64
                {
                    Ok(Scalar::Int64(result as i64))
                } else {
                    Ok(Scalar::Float64(result))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::new(out_dtype, values)
    }

    #[must_use]
    pub fn semantic_eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && self.values.len() == other.values.len()
            && self
                .values
                .iter()
                .zip(&other.values)
                .all(|(left, right)| left.semantic_eq(right))
    }
}

#[cfg(test)]
mod tests {
    use fp_types::{NullKind, Scalar};

    use super::{ArithmeticOp, Column};

    #[test]
    fn reindex_injects_missing_values() {
        let column = Column::from_values(vec![Scalar::Int64(10), Scalar::Int64(20)])
            .expect("column should build");

        let out = column
            .reindex_by_positions(&[Some(1), None, Some(0)])
            .expect("reindex should work");

        assert_eq!(
            out.values(),
            &[
                Scalar::Int64(20),
                Scalar::Null(NullKind::Null),
                Scalar::Int64(10)
            ]
        );
    }

    #[test]
    fn numeric_addition_propagates_missing() {
        let left = Column::from_values(vec![
            Scalar::Int64(1),
            Scalar::Null(NullKind::Null),
            Scalar::Float64(f64::NAN),
        ])
        .expect("left");
        let right = Column::from_values(vec![Scalar::Int64(2), Scalar::Int64(5), Scalar::Int64(3)])
            .expect("right");

        let out = left
            .binary_numeric(&right, ArithmeticOp::Add)
            .expect("add should pass");

        assert_eq!(out.values()[0], Scalar::Float64(3.0));
        assert_eq!(out.values()[1], Scalar::Null(NullKind::NaN));
        assert_eq!(out.values()[2], Scalar::Null(NullKind::NaN));
    }
}
