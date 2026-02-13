#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    Null,
    Bool,
    Int64,
    Float64,
    Utf8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullKind {
    Null,
    NaN,
    NaT,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Scalar {
    Null(NullKind),
    Bool(bool),
    Int64(i64),
    Float64(f64),
    Utf8(String),
}

impl Scalar {
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::Null(_) => DType::Null,
            Self::Bool(_) => DType::Bool,
            Self::Int64(_) => DType::Int64,
            Self::Float64(_) => DType::Float64,
            Self::Utf8(_) => DType::Utf8,
        }
    }

    #[must_use]
    pub fn is_missing(&self) -> bool {
        match self {
            Self::Null(_) => true,
            Self::Float64(v) => v.is_nan(),
            _ => false,
        }
    }

    #[must_use]
    pub fn is_nan(&self) -> bool {
        matches!(self, Self::Null(NullKind::NaN)) || matches!(self, Self::Float64(v) if v.is_nan())
    }

    #[must_use]
    pub fn missing_for_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Float64 => Self::Null(NullKind::NaN),
            DType::Null => Self::Null(NullKind::Null),
            DType::Bool | DType::Int64 | DType::Utf8 => Self::Null(NullKind::Null),
        }
    }

    #[must_use]
    pub fn semantic_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Float64(a), Self::Float64(b)) => (a.is_nan() && b.is_nan()) || (a == b),
            (Self::Null(NullKind::NaN), Self::Float64(v))
            | (Self::Float64(v), Self::Null(NullKind::NaN)) => v.is_nan(),
            _ => self == other,
        }
    }

    pub fn to_f64(&self) -> Result<f64, TypeError> {
        match self {
            Self::Bool(v) => Ok(if *v { 1.0 } else { 0.0 }),
            Self::Int64(v) => Ok(*v as f64),
            Self::Float64(v) => Ok(*v),
            Self::Null(kind) => Err(TypeError::ValueIsMissing { kind: *kind }),
            Self::Utf8(v) => Err(TypeError::NonNumericValue {
                value: v.clone(),
                dtype: DType::Utf8,
            }),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum TypeError {
    #[error("dtype coercion from {left:?} to {right:?} has no compatible common type")]
    IncompatibleDtypes { left: DType, right: DType },
    #[error("cannot cast scalar of dtype {from:?} to {to:?}")]
    InvalidCast { from: DType, to: DType },
    #[error("cannot cast float {value} to int64 without loss")]
    LossyFloatToInt { value: f64 },
    #[error("expected 0/1 for bool cast from int64 but found {value}")]
    InvalidBoolInt { value: i64 },
    #[error("expected 0.0/1.0 for bool cast from float64 but found {value}")]
    InvalidBoolFloat { value: f64 },
    #[error("value {value:?} has non-numeric dtype {dtype:?}")]
    NonNumericValue { value: String, dtype: DType },
    #[error("value is missing ({kind:?})")]
    ValueIsMissing { kind: NullKind },
}

pub fn common_dtype(left: DType, right: DType) -> Result<DType, TypeError> {
    use DType::{Bool, Float64, Int64, Null, Utf8};

    let out = match (left, right) {
        (a, b) if a == b => a,
        (Null, other) | (other, Null) => other,
        (Bool, Int64) | (Int64, Bool) => Int64,
        (Bool, Float64) | (Float64, Bool) => Float64,
        (Int64, Float64) | (Float64, Int64) => Float64,
        (Utf8, Utf8) => Utf8,
        _ => return Err(TypeError::IncompatibleDtypes { left, right }),
    };

    Ok(out)
}

pub fn infer_dtype(values: &[Scalar]) -> Result<DType, TypeError> {
    let mut current = DType::Null;
    for value in values {
        current = common_dtype(current, value.dtype())?;
    }
    Ok(current)
}

pub fn cast_scalar(value: &Scalar, target: DType) -> Result<Scalar, TypeError> {
    let from = value.dtype();
    if from == target || matches!(value, Scalar::Null(_)) {
        return Ok(match value {
            Scalar::Null(_) => Scalar::missing_for_dtype(target),
            _ => value.clone(),
        });
    }

    match target {
        DType::Null => Ok(Scalar::Null(NullKind::Null)),
        DType::Bool => match value {
            Scalar::Bool(v) => Ok(Scalar::Bool(*v)),
            Scalar::Int64(v) => match *v {
                0 => Ok(Scalar::Bool(false)),
                1 => Ok(Scalar::Bool(true)),
                _ => Err(TypeError::InvalidBoolInt { value: *v }),
            },
            Scalar::Float64(v) => {
                if *v == 0.0 {
                    Ok(Scalar::Bool(false))
                } else if *v == 1.0 {
                    Ok(Scalar::Bool(true))
                } else {
                    Err(TypeError::InvalidBoolFloat { value: *v })
                }
            }
            _ => Err(TypeError::InvalidCast { from, to: target }),
        },
        DType::Int64 => match value {
            Scalar::Bool(v) => Ok(Scalar::Int64(i64::from(*v))),
            Scalar::Int64(v) => Ok(Scalar::Int64(*v)),
            Scalar::Float64(v) => {
                if !v.is_finite() || *v != v.trunc() {
                    return Err(TypeError::LossyFloatToInt { value: *v });
                }
                if *v < i64::MIN as f64 || *v > i64::MAX as f64 {
                    return Err(TypeError::LossyFloatToInt { value: *v });
                }
                Ok(Scalar::Int64(*v as i64))
            }
            _ => Err(TypeError::InvalidCast { from, to: target }),
        },
        DType::Float64 => match value {
            Scalar::Bool(v) => Ok(Scalar::Float64(if *v { 1.0 } else { 0.0 })),
            Scalar::Int64(v) => Ok(Scalar::Float64(*v as f64)),
            Scalar::Float64(v) => Ok(Scalar::Float64(*v)),
            _ => Err(TypeError::InvalidCast { from, to: target }),
        },
        DType::Utf8 => match value {
            Scalar::Utf8(v) => Ok(Scalar::Utf8(v.clone())),
            _ => Err(TypeError::InvalidCast { from, to: target }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{DType, NullKind, Scalar, cast_scalar, common_dtype, infer_dtype};

    #[test]
    fn dtype_inference_coerces_numeric_values() {
        let values = vec![Scalar::Bool(true), Scalar::Int64(7), Scalar::Float64(3.5)];
        assert_eq!(
            infer_dtype(&values).expect("dtype should infer"),
            DType::Float64
        );
    }

    #[test]
    fn missing_values_get_target_missing_marker() {
        let missing = Scalar::Null(NullKind::Null);
        let cast = cast_scalar(&missing, DType::Float64).expect("missing casts");
        assert_eq!(cast, Scalar::Null(NullKind::NaN));
    }

    #[test]
    fn semantic_eq_treats_nan_as_equal() {
        let left = Scalar::Float64(f64::NAN);
        let right = Scalar::Null(NullKind::NaN);
        assert!(left.semantic_eq(&right));
    }

    #[test]
    fn common_dtype_rejects_string_numeric_mix() {
        let err = common_dtype(DType::Utf8, DType::Int64).expect_err("must fail");
        assert_eq!(
            err.to_string(),
            "dtype coercion from Utf8 to Int64 has no compatible common type"
        );
    }
}
