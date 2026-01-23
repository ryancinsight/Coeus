//! GPU max/min reduction primitives (placeholder)

use crate::DataType;

/// Max reduction primitive for GPU backend (placeholder)
pub fn max_primitive<T: DataType>(_input: &[T]) -> crate::Result<T>
where
    T: PartialOrd + Copy,
{
    // TODO: Implement GPU-accelerated max reduction
    Err(crate::BackendError::UnsupportedOperation {
        operation: "max_primitive".to_string(),
        backend: "gpu".to_string(),
    })
}

/// Min reduction primitive for GPU backend (placeholder)
pub fn min_primitive<T: DataType>(_input: &[T]) -> crate::Result<T>
where
    T: PartialOrd + Copy,
{
    // TODO: Implement GPU-accelerated min reduction
    Err(crate::BackendError::UnsupportedOperation {
        operation: "min_primitive".to_string(),
        backend: "gpu".to_string(),
    })
}