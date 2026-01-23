//! GPU sparse arithmetic kernels
//!
//! Provides GPU-accelerated element-wise operations for sparse matrices.
//! Currently uses CPU fallbacks; will be replaced with wgpu compute shaders.

use storage::{CsrStorage, Result};
use dtype::DataType;
use std::vec::Vec;

/// GPU sparse addition kernel
/// 
/// Adds two sparse CSR matrices element-wise.
/// For now, uses CPU fallback via dense conversion.
pub fn gpu_sparse_add<T: DataType + Default>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>>
where
    T: Copy + crate::num_traits::Zero + core::ops::Add<Output = T> + PartialEq,
{
    // CPU fallback: convert to dense, add, convert back
    // GPU would use parallel sparse addition kernel
    let lhs_dense = lhs.to_dense()?;
    let rhs_dense = rhs.to_dense()?;
    
    use storage::Storage;
    let lhs_data = lhs_dense.as_slice();
    let rhs_data = rhs_dense.as_slice();
    
    let result_data: Vec<T> = lhs_data.iter()
        .zip(rhs_data.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    
    let result_dense = storage::DenseStorage::from_vec(result_data, lhs.shape_ref().dims())?;
    CsrStorage::from_dense(&result_dense)
}

/// GPU sparse subtraction kernel
pub fn gpu_sparse_sub<T: DataType + Default>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>>
where
    T: Copy + crate::num_traits::Zero + core::ops::Sub<Output = T> + PartialEq,
{
    let lhs_dense = lhs.to_dense()?;
    let rhs_dense = rhs.to_dense()?;
    
    use storage::Storage;
    let lhs_data = lhs_dense.as_slice();
    let rhs_data = rhs_dense.as_slice();
    
    let result_data: Vec<T> = lhs_data.iter()
        .zip(rhs_data.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    
    let result_dense = storage::DenseStorage::from_vec(result_data, lhs.shape_ref().dims())?;
    CsrStorage::from_dense(&result_dense)
}

/// GPU sparse element-wise multiplication kernel
pub fn gpu_sparse_mul<T: DataType + Default>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>>
where
    T: Copy + crate::num_traits::Zero + core::ops::Mul<Output = T> + PartialEq,
{
    let lhs_dense = lhs.to_dense()?;
    let rhs_dense = rhs.to_dense()?;
    
    use storage::Storage;
    let lhs_data = lhs_dense.as_slice();
    let rhs_data = rhs_dense.as_slice();
    
    let result_data: Vec<T> = lhs_data.iter()
        .zip(rhs_data.iter())
        .map(|(&a, &b)| a * b)
        .collect();
    
    let result_dense = storage::DenseStorage::from_vec(result_data, lhs.shape_ref().dims())?;
    CsrStorage::from_dense(&result_dense)
}

/// GPU sparse element-wise division kernel
pub fn gpu_sparse_div<T: DataType + Default>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>>
where
    T: Copy + crate::num_traits::Zero + core::ops::Div<Output = T> + PartialEq,
{
    let lhs_dense = lhs.to_dense()?;
    let rhs_dense = rhs.to_dense()?;
    
    use storage::Storage;
    let lhs_data = lhs_dense.as_slice();
    let rhs_data = rhs_dense.as_slice();
    
    let result_data: Vec<T> = lhs_data.iter()
        .zip(rhs_data.iter())
        .map(|(&a, &b)| a / b)
        .collect();
    
    let result_dense = storage::DenseStorage::from_vec(result_data, lhs.shape_ref().dims())?;
    CsrStorage::from_dense(&result_dense)
}
