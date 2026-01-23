//! GPU ReLU activation primitive (placeholder)

use crate::DataType;

/// ReLU activation primitive for GPU backend (placeholder)
pub fn relu_primitive<T: DataType>(
    _input: &[T],
    _result: &mut [T],
) -> crate::Result<()>
where
    T: PartialOrd + Default + Copy,
{
    // TODO: Implement GPU-accelerated ReLU
    Err(crate::BackendError::UnsupportedOperation {
        operation: "relu_primitive".to_string(),
        backend: "gpu".to_string(),
    })
}