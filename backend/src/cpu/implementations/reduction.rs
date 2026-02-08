use crate::Result;
use dtype::DataType;
use storage::{DenseStorage, Storage};
use crate::cpu::reduction;

pub fn sum_dense<T: DataType>(input: &DenseStorage<T>) -> Result<T> {
    Ok(reduction::sum::sum_primitive(input.as_slice()))
}

pub fn sum_strided<T: DataType>(input: &storage::StridedStorage<T>) -> Result<T> {
    Ok(reduction::sum::sum_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
    ))
}

pub fn mean_dense<T: DataType>(
    input: &DenseStorage<T>,
    _axes: Option<&[usize]>,
) -> Result<DenseStorage<T>> {
    // Use hierarchical primitive
    let mean_val = reduction::mean_primitive(input.as_slice());

    // For now, return scalar result
    DenseStorage::from_vec(vec![mean_val], &[])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn max_dense<T: DataType + PartialOrd>(input: &DenseStorage<T>) -> Result<T> {
    reduction::max::max_primitive(input.as_slice())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Max error: {}", e)))
}

pub fn max_strided<T: DataType + PartialOrd>(input: &storage::StridedStorage<T>) -> Result<T> {
    reduction::max::max_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
    )
}

pub fn min_dense<T: DataType + PartialOrd>(input: &DenseStorage<T>) -> Result<T> {
    reduction::max::min_primitive(input.as_slice())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Min error: {}", e)))
}

pub fn min_strided<T: DataType + PartialOrd>(input: &storage::StridedStorage<T>) -> Result<T> {
    reduction::max::min_strided_primitive(
        input.as_slice(),
        input.shape().dims(),
        input.strides(),
        input.offset(),
    )
}

pub fn argmax_dense<T: DataType + PartialOrd>(input: &DenseStorage<T>) -> Result<usize> {
    input
        .as_slice()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .ok_or_else(|| crate::BackendError::InvalidInput("Empty tensor".to_string()))
}

pub fn argmin_dense<T: DataType + PartialOrd>(input: &DenseStorage<T>) -> Result<usize> {
    input
        .as_slice()
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .ok_or_else(|| crate::BackendError::InvalidInput("Empty tensor".to_string()))
}
