#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use fp_frame::{FrameError, Series};
use fp_runtime::{EvidenceLedger, RuntimePolicy};
use fp_types::Scalar;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SeriesRef(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Expr {
    Series { name: SeriesRef },
    Add { left: Box<Expr>, right: Box<Expr> },
    Literal { value: Scalar },
}

#[derive(Debug, Clone, Default)]
pub struct EvalContext {
    series: BTreeMap<String, Series>,
}

impl EvalContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            series: BTreeMap::new(),
        }
    }

    pub fn insert_series(&mut self, series: Series) {
        self.series.insert(series.name().to_owned(), series);
    }

    #[must_use]
    pub fn get_series(&self, name: &str) -> Option<&Series> {
        self.series.get(name)
    }
}

#[derive(Debug, Error)]
pub enum ExprError {
    #[error("unknown series reference: {0}")]
    UnknownSeries(String),
    #[error("cannot evaluate a pure literal expression without an index anchor")]
    UnanchoredLiteral,
    #[error(transparent)]
    Frame(#[from] FrameError),
}

pub fn evaluate(
    expr: &Expr,
    context: &EvalContext,
    policy: &RuntimePolicy,
    ledger: &mut EvidenceLedger,
) -> Result<Series, ExprError> {
    match expr {
        Expr::Series { name } => context
            .get_series(&name.0)
            .cloned()
            .ok_or_else(|| ExprError::UnknownSeries(name.0.clone())),
        Expr::Add { left, right } => {
            let lhs = evaluate(left, context, policy, ledger)?;
            let rhs = evaluate(right, context, policy, ledger)?;
            lhs.add_with_policy(&rhs, policy, ledger)
                .map_err(ExprError::from)
        }
        Expr::Literal { .. } => Err(ExprError::UnanchoredLiteral),
    }
}

#[cfg(test)]
mod tests {
    use fp_runtime::{EvidenceLedger, RuntimePolicy};
    use fp_types::Scalar;

    use super::{EvalContext, Expr, SeriesRef, evaluate};
    use fp_frame::Series;

    #[test]
    fn expression_add_works_through_series_refs() {
        let a = Series::from_values(
            "a",
            vec![1_i64.into(), 2_i64.into()],
            vec![Scalar::Int64(1), Scalar::Int64(2)],
        )
        .expect("a");
        let b = Series::from_values(
            "b",
            vec![2_i64.into(), 3_i64.into()],
            vec![Scalar::Int64(10), Scalar::Int64(20)],
        )
        .expect("b");

        let mut ctx = EvalContext::new();
        ctx.insert_series(a);
        ctx.insert_series(b);

        let expr = Expr::Add {
            left: Box::new(Expr::Series {
                name: SeriesRef("a".to_owned()),
            }),
            right: Box::new(Expr::Series {
                name: SeriesRef("b".to_owned()),
            }),
        };

        let mut ledger = EvidenceLedger::new();
        let out = evaluate(
            &expr,
            &ctx,
            &RuntimePolicy::hardened(Some(10_000)),
            &mut ledger,
        )
        .expect("eval");
        assert_eq!(out.values()[1], Scalar::Int64(12));
    }
}
