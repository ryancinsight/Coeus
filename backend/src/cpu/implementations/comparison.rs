use crate::Result;
use dtype::{num_traits, DataType};
use storage::{DenseStorage, Storage, StridedStorage};

// Helper for dense comparison
fn compare_dense<T, F>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
    op: F,
) -> Result<DenseStorage<T>>
where
    T: DataType + num_traits::One + num_traits::Zero,
    F: Fn(&T, &T) -> bool,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::BackendError::InvalidInput(
            format!("Shape mismatch for comparison: {:?} vs {:?}", lhs.shape(), rhs.shape())
        ));
    }

    let lhs_slice = lhs.as_slice();
    let rhs_slice = rhs.as_slice();
    let mut result = vec![T::default(); lhs_slice.len()];

    for i in 0..lhs_slice.len() {
        result[i] = if op(&lhs_slice[i], &rhs_slice[i]) {
            T::one()
        } else {
            T::zero()
        };
    }

    DenseStorage::from_vec(result, lhs.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

// Helper for strided comparison
fn compare_strided<T, F>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
    op: F,
) -> Result<StridedStorage<T>>
where
    T: DataType + num_traits::One + num_traits::Zero,
    F: Fn(&T, &T) -> bool,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::BackendError::InvalidInput(
            format!("Shape mismatch for comparison: {:?} vs {:?}", lhs.shape(), rhs.shape())
        ));
    }

    // Since StridedStorage might not be contiguous, we use the iterator.
    // Ideally we would return a StridedStorage. 
    // StridedStorage::from_vec creates a dense (contiguous) storage wrapped in Strided.
    // This effectively "densifies" the result, which is acceptable for comparison output 
    // (usually boolean masks are small or used immediately).
    // Preserving strides for the result is complex if inputs have different strides.
    // For element-wise ops, usually the output is contiguous or matches one of the inputs.
    // Here we'll produce a contiguous result with default strides.

    Err(crate::BackendError::UnsupportedOperation { 
        operation: "compare_strided".to_string(), 
        backend: "cpu".to_string() 
    })
}

// Dense implementations
pub fn eq_dense<T: DataType + num_traits::One + num_traits::Zero>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    compare_dense(lhs, rhs, |a, b| a == b)
}

pub fn ne_dense<T: DataType + num_traits::One + num_traits::Zero>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    compare_dense(lhs, rhs, |a, b| a != b)
}

pub fn gt_dense<T: DataType + PartialOrd + num_traits::One + num_traits::Zero>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    compare_dense(lhs, rhs, |a, b| a > b)
}

pub fn ge_dense<T: DataType + PartialOrd + num_traits::One + num_traits::Zero>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    compare_dense(lhs, rhs, |a, b| a >= b)
}

pub fn lt_dense<T: DataType + PartialOrd + num_traits::One + num_traits::Zero>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    compare_dense(lhs, rhs, |a, b| a < b)
}

pub fn le_dense<T: DataType + PartialOrd + num_traits::One + num_traits::Zero>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    compare_dense(lhs, rhs, |a, b| a <= b)
}

// Strided implementations (Placeholders for now, will fix in next step after reading StridedStorage)
pub fn eq_strided<T: DataType + num_traits::One + num_traits::Zero>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
) -> Result<StridedStorage<T>> {
    compare_strided(lhs, rhs, |a, b| a == b)
}

pub fn ne_strided<T: DataType + num_traits::One + num_traits::Zero>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
) -> Result<StridedStorage<T>> {
    compare_strided(lhs, rhs, |a, b| a != b)
}

pub fn gt_strided<T: DataType + PartialOrd + num_traits::One + num_traits::Zero>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
) -> Result<StridedStorage<T>> {
    compare_strided(lhs, rhs, |a, b| a > b)
}

pub fn ge_strided<T: DataType + PartialOrd + num_traits::One + num_traits::Zero>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
) -> Result<StridedStorage<T>> {
    compare_strided(lhs, rhs, |a, b| a >= b)
}

pub fn lt_strided<T: DataType + PartialOrd + num_traits::One + num_traits::Zero>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
) -> Result<StridedStorage<T>> {
    compare_strided(lhs, rhs, |a, b| a < b)
}

pub fn le_strided<T: DataType + PartialOrd + num_traits::One + num_traits::Zero>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
) -> Result<StridedStorage<T>> {
    compare_strided(lhs, rhs, |a, b| a <= b)
}

// ================== Status Checks ==================

pub fn isnan_dense<T: DataType + num_traits::Float + num_traits::One + num_traits::Zero>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for i in 0..input_slice.len() {
        result[i] = if input_slice[i].is_nan() { T::one() } else { T::zero() };
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn isinf_dense<T: DataType + num_traits::Float + num_traits::One + num_traits::Zero>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for i in 0..input_slice.len() {
        result[i] = if input_slice[i].is_infinite() { T::one() } else { T::zero() };
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn isfinite_dense<T: DataType + num_traits::Float + num_traits::One + num_traits::Zero>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for i in 0..input_slice.len() {
        result[i] = if input_slice[i].is_finite() { T::one() } else { T::zero() };
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

// ================== Logical Operations ==================

pub fn logical_and_dense<T: DataType + num_traits::One + num_traits::Zero>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    compare_dense(lhs, rhs, |a, b| !a.is_zero() && !b.is_zero())
}

pub fn logical_or_dense<T: DataType + num_traits::One + num_traits::Zero>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    compare_dense(lhs, rhs, |a, b| !a.is_zero() || !b.is_zero())
}

pub fn logical_xor_dense<T: DataType + num_traits::One + num_traits::Zero>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    compare_dense(lhs, rhs, |a, b| (!a.is_zero()) ^ (!b.is_zero()))
}

pub fn logical_not_dense<T: DataType + num_traits::One + num_traits::Zero>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    for i in 0..input_slice.len() {
        result[i] = if input_slice[i].is_zero() { T::one() } else { T::zero() };
    }
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}
