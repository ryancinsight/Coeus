//! GPU tanh activation primitive (placeholder)

use crate::DataType;

/// Tanh activation primitive for GPU backend (placeholder)
pub fn tanh_primitive<T: DataType>(
    _input: &[T],
    _result: &mut [T],
) -> crate::Result<()>
where
    T: Copy,
{
    // TODO: Implement GPU-accelerated tanh
    Err(crate::BackendError::UnsupportedOperation {
        operation: "tanh_primitive".to_string(),
        backend: "gpu".to_string(),
    })
}