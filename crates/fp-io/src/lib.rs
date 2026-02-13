#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use csv::{ReaderBuilder, WriterBuilder};
use fp_columnar::{Column, ColumnError};
use fp_frame::{DataFrame, FrameError};
use fp_index::Index;
use fp_types::{NullKind, Scalar};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("csv input has no headers")]
    MissingHeaders,
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error(transparent)]
    Column(#[from] ColumnError),
    #[error(transparent)]
    Frame(#[from] FrameError),
}

pub fn read_csv_str(input: &str) -> Result<DataFrame, IoError> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(input.as_bytes());

    let headers = reader.headers().cloned().map_err(IoError::from)?;

    if headers.is_empty() {
        return Err(IoError::MissingHeaders);
    }

    let mut columns = headers
        .iter()
        .map(|name| (name.to_owned(), Vec::<Scalar>::new()))
        .collect::<BTreeMap<_, _>>();

    let mut row_count: i64 = 0;
    for row in reader.records() {
        let record = row?;
        for (idx, header) in headers.iter().enumerate() {
            let field = record.get(idx).unwrap_or_default();
            if let Some(values) = columns.get_mut(header) {
                values.push(parse_scalar(field));
            }
        }
        row_count += 1;
    }

    let mut out_columns = BTreeMap::new();
    for (name, values) in columns {
        out_columns.insert(name, Column::from_values(values)?);
    }

    let index = Index::from_i64((0..row_count).collect());
    Ok(DataFrame::new(index, out_columns)?)
}

pub fn write_csv_string(frame: &DataFrame) -> Result<String, IoError> {
    let mut writer = WriterBuilder::new().from_writer(Vec::new());

    let headers = frame.columns().keys().cloned().collect::<Vec<_>>();
    writer.write_record(&headers)?;

    for row_idx in 0..frame.index().len() {
        let row = headers
            .iter()
            .map(|name| {
                frame
                    .column(name)
                    .and_then(|column| column.value(row_idx))
                    .map_or_else(String::new, scalar_to_csv)
            })
            .collect::<Vec<_>>();
        writer.write_record(&row)?;
    }

    let bytes = writer.into_inner().map_err(|err| err.into_error())?;
    Ok(String::from_utf8(bytes)?)
}

fn parse_scalar(field: &str) -> Scalar {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return Scalar::Null(NullKind::Null);
    }

    if let Ok(value) = trimmed.parse::<i64>() {
        return Scalar::Int64(value);
    }
    if let Ok(value) = trimmed.parse::<f64>() {
        return Scalar::Float64(value);
    }
    if let Ok(value) = trimmed.parse::<bool>() {
        return Scalar::Bool(value);
    }

    Scalar::Utf8(trimmed.to_owned())
}

fn scalar_to_csv(scalar: &Scalar) -> String {
    match scalar {
        Scalar::Null(_) => String::new(),
        Scalar::Bool(v) => v.to_string(),
        Scalar::Int64(v) => v.to_string(),
        Scalar::Float64(v) => {
            if v.is_nan() {
                String::new()
            } else {
                v.to_string()
            }
        }
        Scalar::Utf8(v) => v.clone(),
    }
}

#[cfg(test)]
mod tests {
    use fp_types::{NullKind, Scalar};

    use super::{read_csv_str, write_csv_string};

    #[test]
    fn csv_round_trip_preserves_null_and_numeric_shape() {
        let input = "id,value\n1,10\n2,\n3,3.5\n";
        let frame = read_csv_str(input).expect("read");
        let value_col = frame.column("value").expect("value");

        assert_eq!(value_col.values()[1], Scalar::Null(NullKind::NaN));

        let out = write_csv_string(&frame).expect("write");
        assert!(out.contains("id,value"));
        assert!(out.contains("3,3.5"));
    }
}
