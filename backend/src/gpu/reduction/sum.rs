//! GPU sum reduction primitive (placeholder)

use crate::DataType;

/// Sum reduction primitive for GPU backend (placeholder)
pub fn sum_primitive<T: DataType>(_input: &[T]) -> T
where
    T: core::ops::Add<Output = T> + Default + Copy,
{
    // TODO: Implement GPU-accelerated sum reduction
    // For now, return zero as placeholder
    T::default()
}