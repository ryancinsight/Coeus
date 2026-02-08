use crate::Result;
use dtype::{num_traits, DataType};
use storage::{CsrStorage, DenseStorage, Storage, StridedStorage};
use crate::cpu::{arithmetic, sparse_kernels};

pub fn add_dense<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let lhs_slice = lhs.as_slice();
    let rhs_slice = rhs.as_slice();
    let mut result = vec![T::default(); lhs_slice.len()];

    arithmetic::add_primitive(lhs_slice, rhs_slice, &mut result)?;

    DenseStorage::from_vec(result, lhs.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn sub_dense<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let lhs_slice = lhs.as_slice();
    let rhs_slice = rhs.as_slice();
    let mut result = vec![T::default(); lhs_slice.len()];

    arithmetic::sub_primitive(lhs_slice, rhs_slice, &mut result)?;

    DenseStorage::from_vec(result, lhs.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn mul_dense<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let lhs_slice = lhs.as_slice();
    let rhs_slice = rhs.as_slice();
    let mut result = vec![T::default(); lhs_slice.len()];

    arithmetic::mul_primitive(lhs_slice, rhs_slice, &mut result)?;

    DenseStorage::from_vec(result, lhs.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn div_dense<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let lhs_slice = lhs.as_slice();
    let rhs_slice = rhs.as_slice();
    let mut result = vec![T::default(); lhs_slice.len()];

    arithmetic::div_primitive(lhs_slice, rhs_slice, &mut result)?;

    DenseStorage::from_vec(result, lhs.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn add_strided<T: DataType>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
) -> Result<StridedStorage<T>> {
    let mut result_data = vec![T::default(); lhs.shape().size()];
    arithmetic::add_strided_primitive(
        lhs.as_slice(),
        lhs.shape().dims(),
        lhs.strides(),
        lhs.offset(),
        rhs.as_slice(),
        rhs.shape().dims(),
        rhs.strides(),
        rhs.offset(),
        &mut result_data,
    )?;

    StridedStorage::new(result_data, lhs.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn mul_strided<T: DataType>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
) -> Result<StridedStorage<T>> {
    let mut result_data = vec![T::default(); lhs.shape().size()];
    arithmetic::mul_strided_primitive(
        lhs.as_slice(),
        lhs.shape().dims(),
        lhs.strides(),
        lhs.offset(),
        rhs.as_slice(),
        rhs.shape().dims(),
        rhs.strides(),
        rhs.offset(),
        &mut result_data,
    )?;

    StridedStorage::new(result_data, lhs.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn sub_strided<T: DataType>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
) -> Result<StridedStorage<T>> {
    let mut result_data = vec![T::default(); lhs.shape().size()];
    arithmetic::sub_strided_primitive(
        lhs.as_slice(),
        lhs.shape().dims(),
        lhs.strides(),
        lhs.offset(),
        rhs.as_slice(),
        rhs.shape().dims(),
        rhs.strides(),
        rhs.offset(),
        &mut result_data,
    )?;

    StridedStorage::new(result_data, lhs.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn div_strided<T: DataType>(
    lhs: &StridedStorage<T>,
    rhs: &StridedStorage<T>,
) -> Result<StridedStorage<T>> {
    let mut result_data = vec![T::default(); lhs.shape().size()];
    arithmetic::div_strided_primitive(
        lhs.as_slice(),
        lhs.shape().dims(),
        lhs.strides(),
        lhs.offset(),
        rhs.as_slice(),
        rhs.shape().dims(),
        rhs.strides(),
        rhs.offset(),
        &mut result_data,
    )?;

    StridedStorage::new(result_data, lhs.shape().dims())
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn add_csr<T: DataType>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>> {
    if lhs.shape() != rhs.shape() {
        return Err(crate::BackendError::InvalidInput(
            format!("Shapes must match for CSR addition: {:?} vs {:?}", lhs.shape(), rhs.shape())
        ));
    }

    let (data, indices, indptr) = sparse_kernels::csr_add_csr_kernel(
        lhs.data(),
        lhs.indices(),
        lhs.indptr(),
        rhs.data(),
        rhs.indices(),
        rhs.indptr(),
        lhs.shape().dims()[0],
    )?;

    CsrStorage::new(data, indices, indptr, lhs.shape().dims())
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn sub_csr<T: DataType>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>> {
    if lhs.shape() != rhs.shape() {
        return Err(crate::BackendError::InvalidInput(
            format!("Shapes must match for CSR subtraction: {:?} vs {:?}", lhs.shape(), rhs.shape())
        ));
    }

    let (data, indices, indptr) = sparse_kernels::csr_sub_csr_kernel(
        lhs.data(),
        lhs.indices(),
        lhs.indptr(),
        rhs.data(),
        rhs.indices(),
        rhs.indptr(),
        lhs.shape().dims()[0],
    )?;

    CsrStorage::new(data, indices, indptr, lhs.shape().dims())
        .map_err(|e| crate::BackendError::StorageError { source: e })
}

pub fn mul_csr<T: DataType>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>> {
    if lhs.shape() != rhs.shape() {
        return Err(crate::BackendError::InvalidInput(
            format!("Shapes must match for CSR multiplication: {:?} vs {:?}", lhs.shape(), rhs.shape())
        ));
    }

    let (data, indices, indptr) = sparse_kernels::csr_mul_csr_kernel(
        lhs.data(),
        lhs.indices(),
        lhs.indptr(),
        rhs.data(),
        rhs.indices(),
        rhs.indptr(),
        lhs.shape().dims()[0],
    )?;

    CsrStorage::new(data, indices, indptr, lhs.shape().dims())
        .map_err(|e| crate::BackendError::StorageError { source: e })
}
