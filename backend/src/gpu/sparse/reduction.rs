//! GPU sparse reduction kernels
//!
//! Provides GPU-accelerated reduction operations for sparse matrices.
//! Currently uses CPU fallbacks; will be replaced with wgpu parallel reduction.

use storage::CsrStorage;
use dtype::DataType;

/// GPU sparse sum reduction kernel
pub fn gpu_sparse_sum<T: DataType>(storage: &CsrStorage<T>) -> T
where
    T: Copy + Default + core::ops::Add<Output = T>,
{
    // CPU fallback: iterate over non-zero data
    storage.data().iter().copied().fold(T::default(), |acc, x| acc + x)
}

/// GPU sparse max reduction kernel
pub fn gpu_sparse_max<T: DataType>(storage: &CsrStorage<T>) -> Option<T>
where
    T: Copy + PartialOrd,
{
    // CPU fallback: find max in non-zero data
    storage.data().iter().copied().reduce(|a, b| if a > b { a } else { b })
}

/// GPU sparse min reduction kernel
pub fn gpu_sparse_min<T: DataType>(storage: &CsrStorage<T>) -> Option<T>
where
    T: Copy + PartialOrd,
{
    // CPU fallback: find min in non-zero data
    storage.data().iter().copied().reduce(|a, b| if a < b { a } else { b })
}

/// GPU sparse mean reduction kernel
/// 
/// Returns mean as f64 of non-zero elements.
pub fn gpu_sparse_mean<T: DataType>(storage: &CsrStorage<T>) -> f64
where
    T: Copy + Into<f64>,
{
    let nnz = storage.nnz();
    if nnz == 0 {
        return 0.0;
    }
    let sum: f64 = storage.data().iter().map(|&x| x.into()).sum();
    sum / (nnz as f64)
}
