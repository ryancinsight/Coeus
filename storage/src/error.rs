//! Error types for storage operations

extern crate alloc;
use core::fmt;

/// Errors that can occur during storage operations
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StorageError {
    /// Shape and data size mismatch
    ShapeMismatch {
        /// Expected size from shape
        expected: usize,
        /// Actual data size
        actual: usize,
    },

    /// Invalid shape specification
    InvalidShape {
        /// Description of what makes the shape invalid
        reason: &'static str,
    },

    /// Invalid stride specification
    InvalidStride {
        /// Description of what makes the stride invalid
        reason: &'static str,
    },

    /// Index out of bounds
    IndexOutOfBounds {
        /// The invalid index
        index: usize,
        /// Maximum valid index
        bound: usize,
    },

    /// Broadcasting error - incompatible shapes
    BroadcastError {
        /// First shape
        shape_a: alloc::vec::Vec<usize>,
        /// Second shape
        shape_b: alloc::vec::Vec<usize>,
        /// Dimension where incompatibility occurs
        dimension: usize,
    },
}

#[cfg(feature = "std")]
impl std::error::Error for StorageError {}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape requires {expected} elements, but data has {actual}"
                )
            }
            Self::InvalidShape { reason } => {
                write!(f, "Invalid shape: {reason}")
            }
            Self::InvalidStride { reason } => {
                write!(f, "Invalid stride: {reason}")
            }
            Self::IndexOutOfBounds { index, bound } => {
                write!(f, "Index {index} out of bounds for size {bound}")
            }
            Self::BroadcastError {
                shape_a,
                shape_b,
                dimension,
            } => {
                write!(
                    f,
                    "Incompatible shapes for broadcasting: {shape_a:?} and {shape_b:?} at dimension {dimension}"
                )
            }
        }
    }
}
