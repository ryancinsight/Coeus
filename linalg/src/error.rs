use coeus_error::{Error, LinalgError};

// We can define internal helpers that map to coeus_error::LinalgError variants
// for convenience.

pub fn singular_matrix() -> Error {
    Error::Linalg(LinalgError::SingularMatrix("Matrix is singular".into()))
}

pub fn not_square(rows: usize, cols: usize) -> Error {
    Error::Linalg(LinalgError::NotSquare(format!("{}x{}", rows, cols)))
}

pub fn dimension_mismatch(msg: impl Into<String>) -> Error {
    Error::Linalg(LinalgError::DimensionMismatch(msg.into()))
}
