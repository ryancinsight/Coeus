//! GPU sparse kernels
//!
//! This module provides GPU-accelerated operations for sparse matrices.
//! All operations use the CSR format for optimal GPU performance.

pub mod arithmetic;
pub mod reduction;
pub mod spmv;
use storage::DenseStorage;
use crate::Result;
use dtype::DataType;

/// GPU sparse kernel dispatcher trait
pub trait GpuSparseKernels<T: DataType> {
    /// GPU-accelerated sparse matrix-vector multiplication
    fn gpu_spmv(&self, vector: &DenseStorage<T>) -> Result<DenseStorage<T>>
    where
        T: Copy + crate::num_traits::Zero + core::ops::Add<Output = T> + core::ops::Mul<Output = T>;
    
    /// GPU-accelerated sparse sum reduction
    fn gpu_sparse_sum(&self) -> Result<T>
    where
        T: Copy + Default + core::ops::Add<Output = T>;
    
    /// GPU-accelerated sparse max reduction
    fn gpu_sparse_max(&self) -> Option<T>
    where
        T: Copy + PartialOrd;
    
    /// GPU-accelerated sparse min reduction
    fn gpu_sparse_min(&self) -> Option<T>
    where
        T: Copy + PartialOrd;
}
