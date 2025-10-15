//! Error types for dtype operations
//!
//! Provides typed error handling for all dtype-related operations,
//! ensuring safe and predictable error propagation.

use core::fmt;

/// Errors that can occur during dtype operations
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DtypeError {
    /// Error when casting between incompatible types
    CastError {
        /// Source dtype
        from: crate::Dtype,
        /// Target dtype
        to: crate::Dtype,
        /// String representation of the value that couldn't be cast
        value: &'static str,
    },

    /// Error for numerical operations that overflow
    OverflowError {
        /// Operation that caused overflow
        operation: &'static str,
        /// Dtype of the operands
        dtype: crate::Dtype,
    },

    /// Error for division by zero
    DivisionByZero {
        /// Dtype of the operands
        dtype: crate::Dtype,
    },

    /// Error for invalid numerical operations (e.g., sqrt of negative)
    InvalidOperation {
        /// Operation that failed
        operation: &'static str,
        /// Reason for failure
        reason: &'static str,
        /// Dtype of the operand
        dtype: crate::Dtype,
    },

    /// Error for operations on incompatible dtypes
    IncompatibleTypes {
        /// Left operand dtype
        left: crate::Dtype,
        /// Right operand dtype
        right: crate::Dtype,
        /// Operation being performed
        operation: &'static str,
    },
}

#[cfg(feature = "std")]
impl std::error::Error for DtypeError {}

impl fmt::Display for DtypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CastError { from, to, value } => {
                write!(f, "Cannot cast value '{value}' from {from} to {to}")
            }
            Self::OverflowError { operation, dtype } => {
                write!(
                    f,
                    "Arithmetic overflow in {operation} operation for {dtype}"
                )
            }
            Self::DivisionByZero { dtype } => {
                write!(f, "Division by zero for {dtype}")
            }
            Self::InvalidOperation {
                operation,
                reason,
                dtype,
            } => {
                write!(f, "Invalid {operation} operation for {dtype}: {reason}")
            }
            Self::IncompatibleTypes {
                left,
                right,
                operation,
            } => {
                write!(
                    f,
                    "Incompatible types for {operation} operation: {left} and {right}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl From<DtypeError> for std::io::Error {
    fn from(error: DtypeError) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, error)
    }
}

// Tests will be added with proper std/alloc handling
