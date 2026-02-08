use crate::Result;
use dtype::DataType;
use dtype::num_traits;
use storage::{DenseStorage, Storage, CsrStorage};
use crate::cpu::sparse_kernels;

pub fn spmv_csr<T: DataType>(
    data: &[T],
    indices: &[usize],
    indptr: &[usize],
    vector: &[T],
    num_rows: usize,
    _num_cols: usize,
) -> Result<Vec<T>> {
    let mut result = vec![T::default(); num_rows];
    sparse_kernels::spmv_csr_kernel(
        data,
        indices,
        indptr,
        vector,
        &mut result,
        num_rows,
    )?;
    Ok(result)
}

pub fn spmm_csr<T: DataType>(
    data: &[T],
    indices: &[usize],
    indptr: &[usize],
    other: &DenseStorage<T>,
    num_rows: usize,
    _num_cols: usize,
) -> Result<Vec<T>> {
    let dense_cols = other.shape().dims().get(1).copied().unwrap_or(1);
    let mut result = vec![T::default(); num_rows * dense_cols];
    sparse_kernels::spmm_csr_dense_kernel(
        data,
        indices,
        indptr,
        other.as_slice(),
        dense_cols,
        &mut result,
        num_rows,
    )?;
    Ok(result)
}

pub fn coo_matmul_sparse<T: DataType>(
    _lhs_data: &[T],
    _lhs_row: &[usize],
    _lhs_col: &[usize],
    _rhs_data: &[T],
    _rhs_row: &[usize],
    _rhs_col: &[usize],
    m: usize,
    _k: usize,
    n: usize,
) -> Result<CsrStorage<T>> {
    // Placeholder implementation
    CsrStorage::empty(&[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn coo_matmul_dense<T: DataType>(
    _lhs_data: &[T],
    _lhs_row: &[usize],
    _lhs_col: &[usize],
    _rhs: &DenseStorage<T>,
    m: usize,
    _k: usize,
    n: usize,
) -> Result<DenseStorage<T>> {
    // Placeholder implementation
    DenseStorage::from_vec(vec![T::default(); m * n], &[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn coo_add_sparse<T: DataType>(
    lhs_data: &[T],
    lhs_row: &[usize],
    lhs_col: &[usize],
    rhs_data: &[T],
    rhs_row: &[usize],
    rhs_col: &[usize],
    m: usize,
    n: usize,
) -> Result<CsrStorage<T>> {
    let lhs_coo = storage::CooStorage::new(lhs_data.to_vec(), lhs_row.to_vec(), lhs_col.to_vec(), &[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
    let rhs_coo = storage::CooStorage::new(rhs_data.to_vec(), rhs_row.to_vec(), rhs_col.to_vec(), &[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
    
    let lhs_csr = lhs_coo.to_csr().map_err(|e| crate::BackendError::StorageError { source: e })?;
    let rhs_csr = rhs_coo.to_csr().map_err(|e| crate::BackendError::StorageError { source: e })?;
    
    let (data, indices, indptr) = sparse_kernels::csr_add_csr_kernel(
        lhs_csr.data(),
        lhs_csr.indices(),
        lhs_csr.indptr(),
        rhs_csr.data(),
        rhs_csr.indices(),
        rhs_csr.indptr(),
        m,
    )?;

    CsrStorage::new(data, indices, indptr, &[m, n])
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn coo_mul_sparse<T: DataType>(
    lhs_data: &[T],
    lhs_row: &[usize],
    lhs_col: &[usize],
    rhs_data: &[T],
    rhs_row: &[usize],
    rhs_col: &[usize],
    m: usize,
    n: usize,
) -> Result<CsrStorage<T>> {
    let lhs_coo = storage::CooStorage::new(lhs_data.to_vec(), lhs_row.to_vec(), lhs_col.to_vec(), &[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
    let rhs_coo = storage::CooStorage::new(rhs_data.to_vec(), rhs_row.to_vec(), rhs_col.to_vec(), &[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
    
    let lhs_csr = lhs_coo.to_csr().map_err(|e| crate::BackendError::StorageError { source: e })?;
    let rhs_csr = rhs_coo.to_csr().map_err(|e| crate::BackendError::StorageError { source: e })?;
    
    let (data, indices, indptr) = sparse_kernels::csr_mul_csr_kernel(
        lhs_csr.data(),
        lhs_csr.indices(),
        lhs_csr.indptr(),
        rhs_csr.data(),
        rhs_csr.indices(),
        rhs_csr.indptr(),
        m,
    )?;

    CsrStorage::new(data, indices, indptr, &[m, n])
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn add_dense_csr<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<DenseStorage<T>> {
    let (rows, cols) = rhs.dims();
    let data = sparse_kernels::add_dense_csr_kernel(
        lhs.as_slice(),
        rhs.data(),
        rhs.indices(),
        rhs.indptr(),
        rows,
        cols,
    )?;
    DenseStorage::from_vec(data, &[rows, cols])
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn mul_dense_csr<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<DenseStorage<T>> {
    let (rows, cols) = rhs.dims();
    let out_data = sparse_kernels::mul_dense_csr_kernel(
        lhs.as_slice(),
        rhs.data(),
        rhs.indices(),
        rhs.indptr(),
        rows,
        cols,
    )?;
    // Returning CSR result but Backend trait wants Dense for now to match unified API.
    // In future, optimize to return CSR if possible.
    // For now, densify the sparse result.
    let csr_res = CsrStorage::new(out_data, rhs.indices().to_vec(), rhs.indptr().to_vec(), &[rows, cols])
        .map_err(|e| crate::BackendError::StorageError { source: e })?;
    csr_res.to_dense()
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn matmul_dense_csr<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<DenseStorage<T>> {
    let m = lhs.shape().dims()[0];
    let k = lhs.shape().dims()[1];
    let n = rhs.dims().1;
    let mut result_data = vec![T::default(); m * n];
    sparse_kernels::matmul_dense_csr_kernel(
        lhs.as_slice(),
        rhs.data(),
        rhs.indices(),
        rhs.indptr(),
        m,
        k,
        n,
        &mut result_data,
    )?;
    DenseStorage::from_vec(result_data, &[m, n])
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn matmul_csr<T: DataType>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>>
where 
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + Copy + PartialEq,
{
    let (m, k) = lhs.dims();
    let (k2, n) = rhs.dims();
    
    if k != k2 {
        return Err(crate::BackendError::InvalidInput(format!("Shape mismatch for matmul: ({}, {}) vs ({}, {})", m, k, k2, n)));
    }

    let (data, indices, indptr) = sparse_kernels::csr_matmul_csr_kernel(
        lhs.data(),
        lhs.indices(),
        lhs.indptr(),
        rhs.data(),
        rhs.indices(),
        rhs.indptr(),
        m,
        k,
        n,
    )?;

    CsrStorage::new(data, indices, indptr, &[m, n])
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn addmm_csr<T: DataType>(
    input: &CsrStorage<T>,
    mat1: &CsrStorage<T>,
    mat2: &CsrStorage<T>,
    beta: T,
    alpha: T,
) -> Result<CsrStorage<T>>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + Copy + PartialEq,
{
    // beta * input + alpha * (mat1 @ mat2)
    
    // 1. Matmul
    let prod = matmul_csr(mat1, mat2)?;

    // 2. Scale product if alpha != 1
    // Note: CsrStorage is immutable here, we need to create new or clone if modifying.
    // Optimizing: if alpha is 1, use prod directly.
    
    let prod_scaled = if alpha == T::default() + T::default() + num_traits::One::one() { // alpha == 1
         prod
    } else {
         let mut data = prod.data().to_vec();
         for d in &mut data { *d = *d * alpha; }
         CsrStorage::new(data, prod.indices().to_vec(), prod.indptr().to_vec(), prod.shape().dims())
            .map_err(|e| crate::BackendError::StorageError { source: e })?
    };

    // 3. Scale input if beta != 1
    let input_scaled = if beta == T::default() + T::default() + num_traits::One::one() {
         input.clone() // Need clone to return new storage
    } else {
         let mut data = input.data().to_vec();
         for d in &mut data { *d = *d * beta; }
         CsrStorage::new(data, input.indices().to_vec(), input.indptr().to_vec(), input.shape().dims())
             .map_err(|e| crate::BackendError::StorageError { source: e })?
    };

    // 4. Add
    // Use kernel directly
    let (m, n) = input.dims();
    let (data, indices, indptr) = sparse_kernels::csr_add_csr_kernel(
        input_scaled.data(),
        input_scaled.indices(),
        input_scaled.indptr(),
        prod_scaled.data(),
        prod_scaled.indices(),
        prod_scaled.indptr(),
        m,
    )?;

    CsrStorage::new(data, indices, indptr, &[m, n])
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn addmv_csr<T: DataType>(
    input: &CsrStorage<T>,
    mat: &CsrStorage<T>,
    vec: &CsrStorage<T>, // Vector as CSR matrix (col vector)
    beta: T,
    alpha: T,
) -> Result<CsrStorage<T>>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + Copy + PartialEq,
{
    // Reuse addmm because generic CSR handles vectors as matrices
    addmm_csr(input, mat, vec, beta, alpha)
}

pub fn coo_sub_sparse<T: DataType>(
    lhs_data: &[T],
    lhs_row: &[usize],
    lhs_col: &[usize],
    rhs_data: &[T],
    rhs_row: &[usize],
    rhs_col: &[usize],
    m: usize,
    n: usize,
) -> Result<CsrStorage<T>> 
where
    T: core::ops::Sub<Output = T> + Copy + Default,
{
    let lhs_coo = storage::CooStorage::new(lhs_data.to_vec(), lhs_row.to_vec(), lhs_col.to_vec(), &[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
    let rhs_coo = storage::CooStorage::new(rhs_data.to_vec(), rhs_row.to_vec(), rhs_col.to_vec(), &[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
    
    let lhs_csr = lhs_coo.to_csr().map_err(|e| crate::BackendError::StorageError { source: e })?;
    let rhs_csr = rhs_coo.to_csr().map_err(|e| crate::BackendError::StorageError { source: e })?;
    
    let (data, indices, indptr) = sparse_kernels::csr_sub_csr_kernel(
        lhs_csr.data(),
        lhs_csr.indices(),
        lhs_csr.indptr(),
        rhs_csr.data(),
        rhs_csr.indices(),
        rhs_csr.indptr(),
        m,
    )?;

    CsrStorage::new(data, indices, indptr, &[m, n])
        .map_err(|e| crate::BackendError::StorageError { source: e })
}
