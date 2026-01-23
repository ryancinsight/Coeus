//! GPU element-wise subtraction primitive (placeholder)

use crate::DataType;

/// Element-wise subtraction primitive for GPU backend (placeholder)
pub fn sub_primitive<T: DataType>(
    _lhs: &[T],
    _rhs: &[T],
    _result: &mut [T],
) -> crate::Result<()>
where
    T: core::ops::Sub<Output = T> + Copy,
{
    // TODO: Implement GPU-accelerated subtraction
    Err(crate::BackendError::UnsupportedOperation {
        operation: "sub_primitive".to_string(),
        backend: "gpu".to_string(),
    })
}