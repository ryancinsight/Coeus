use leto::{Array, LetoError, Result, Storage};
use leto_ops::RealScalar;

use super::{shape_n, MAX_DISPATCH_RANK};

fn from_shape_fn_values_n<T: Clone, F, const N: usize>(shape: &[usize], f: &F) -> Result<Vec<T>>
where
    F: Fn(&[usize]) -> T,
{
    let values = Array::<T, _, N>::from_shape_fn(shape_n::<N>(shape)?, |index| f(&index));
    Ok(values.storage().as_slice().to_vec())
}

/// Generate C-contiguous row-major values for a coeus dynamic-rank shape,
/// dispatched to Leto's const-rank coordinate generator.
pub fn from_shape_fn_values<T: Clone, F>(shape: &[usize], f: F) -> Result<Vec<T>>
where
    F: Fn(&[usize]) -> T,
{
    match shape.len() {
        1 => from_shape_fn_values_n::<T, F, 1>(shape, &f),
        2 => from_shape_fn_values_n::<T, F, 2>(shape, &f),
        3 => from_shape_fn_values_n::<T, F, 3>(shape, &f),
        4 => from_shape_fn_values_n::<T, F, 4>(shape, &f),
        5 => from_shape_fn_values_n::<T, F, 5>(shape, &f),
        6 => from_shape_fn_values_n::<T, F, 6>(shape, &f),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn uniform_values_n<T: RealScalar, const N: usize>(
    shape: &[usize],
    low: T,
    high: T,
    seed: u64,
) -> Result<Vec<T>> {
    let values = leto_ops::uniform_with_seed(shape_n::<N>(shape)?, low, high, seed)?;
    Ok(values.storage().as_slice().to_vec())
}

/// Deterministic uniform initialization values for a coeus dynamic-rank shape,
/// dispatched to the matching monomorphized leto random constructor.
pub fn uniform_values<T: RealScalar>(
    shape: &[usize],
    low: T,
    high: T,
    seed: u64,
) -> Result<Vec<T>> {
    match shape.len() {
        1 => uniform_values_n::<T, 1>(shape, low, high, seed),
        2 => uniform_values_n::<T, 2>(shape, low, high, seed),
        3 => uniform_values_n::<T, 3>(shape, low, high, seed),
        4 => uniform_values_n::<T, 4>(shape, low, high, seed),
        5 => uniform_values_n::<T, 5>(shape, low, high, seed),
        6 => uniform_values_n::<T, 6>(shape, low, high, seed),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn normal_values_n<T: RealScalar, const N: usize>(
    shape: &[usize],
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<Vec<T>> {
    let values = leto_ops::normal_with_seed(shape_n::<N>(shape)?, mean, std_dev, seed)?;
    Ok(values.storage().as_slice().to_vec())
}

/// Deterministic normal initialization values for a coeus dynamic-rank shape,
/// dispatched to the matching monomorphized leto random constructor.
pub fn normal_values<T: RealScalar>(
    shape: &[usize],
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<Vec<T>> {
    match shape.len() {
        1 => normal_values_n::<T, 1>(shape, mean, std_dev, seed),
        2 => normal_values_n::<T, 2>(shape, mean, std_dev, seed),
        3 => normal_values_n::<T, 3>(shape, mean, std_dev, seed),
        4 => normal_values_n::<T, 4>(shape, mean, std_dev, seed),
        5 => normal_values_n::<T, 5>(shape, mean, std_dev, seed),
        6 => normal_values_n::<T, 6>(shape, mean, std_dev, seed),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}
