//! Dense conversion functions

use super::core::DenseStorage;
use crate::{DataType, Result};

impl<T: DataType> DenseStorage<T> {
    /// Convert to dense (identity operation)
    pub fn to_dense(&self) -> Result<DenseStorage<T>> {
        Ok(self.clone())
    }

    /// Convert to CSR format
    pub fn to_csr(&self) -> Result<crate::sparse::CsrStorage<T>>
    where
        T: num_traits::Zero + PartialEq,
    {
        crate::sparse::CsrStorage::from_dense(self)
    }

    /// Convert to CSC format
    pub fn to_csc(&self) -> Result<crate::sparse::CscStorage<T>>
    where
        T: num_traits::Zero + PartialEq,
    {
        crate::sparse::CscStorage::from_dense(self)
    }

    /// Convert to COO format
    pub fn to_coo(&self) -> Result<crate::sparse::CooStorage<T>>
    where
        T: num_traits::Zero + PartialEq,
    {
        crate::sparse::CooStorage::from_dense(self)
    }
}
