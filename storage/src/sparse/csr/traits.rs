//! CSR trait implementations
//!
//! Implements Storage, StorageToDense, StorageFromVec, AsAny traits.

use super::core::CsrStorage;
use crate::{
    AsAny, DataType, DenseStorage, Result, Shape, Storage, StorageError, StorageFormat,
    StorageFromVec, StorageToDense,
};
use alloc::vec;
use core::fmt;

impl<T: DataType> Storage<T> for CsrStorage<T> {
    fn len(&self) -> usize {
        self.shape.dims().iter().product()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn format(&self) -> StorageFormat {
        StorageFormat::Csr
    }

    fn strides(&self) -> &[usize] {
        &[]
    }

    fn is_contiguous(&self) -> bool {
        false
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn as_storage_ref(&self) -> &Self {
        self
    }

    fn full(dims: &[usize], value: T) -> Result<Self>
    where
        Self: Sized,
    {
        if dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "CSR storage requires 2D shape",
            });
        }

        let rows = dims[0];
        let cols = dims[1];
        let total_elements = rows * cols;

        if value == T::zero() {
            Self::empty(dims)
        } else {
            let data = vec![value; total_elements];
            let indices: alloc::vec::Vec<usize> =
                (0..cols).cycle().take(total_elements).collect();
            let indptr: alloc::vec::Vec<usize> = (0..=rows).map(|i| i * cols).collect();
            Self::new(data, indices, indptr, dims)
        }
    }

    fn map_structure<F>(&self, f: F) -> Result<Self>
    where
        Self: Sized,
        F: FnMut(T) -> T,
    {
        let data = self.data.iter().cloned().map(f).collect();
        Ok(Self {
            data,
            indices: self.indices.clone(),
            indptr: self.indptr.clone(),
            shape: self.shape.clone(),
        })
    }
}

impl<T: DataType> StorageToDense<T> for CsrStorage<T>
where
    T: num_traits::Zero,
{
    fn to_dense(&self) -> Result<DenseStorage<T>> {
        self.to_dense()
    }
}

impl<T: DataType> StorageFromVec<T> for CsrStorage<T>
where
    T: num_traits::Zero + PartialEq,
{
    fn from_vec(data: alloc::vec::Vec<T>, shape: &[usize]) -> Result<Self> {
        let dense = DenseStorage::from_vec(data, shape)?;
        Self::from_dense(&dense)
    }

    fn zeros(dims: &[usize]) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Zero,
    {
        Self::empty(dims)
    }

    fn ones(dims: &[usize]) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::One,
    {
        Self::full(dims, T::one())
    }
}

impl<T: DataType + 'static> AsAny for CsrStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: DataType + fmt::Display + num_traits::Zero> fmt::Display for CsrStorage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.dims();
        writeln!(f, "CsrStorage {}x{} (nnz={})", rows, cols, self.nnz())?;

        if rows <= 10 && cols <= 10 {
            for row in 0..rows {
                write!(f, "[")?;
                for col in 0..cols {
                    if col > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.get(row, col))?;
                }
                writeln!(f, "]")?;
            }
        } else {
            writeln!(f, "Matrix too large to display")?;
        }

        Ok(())
    }
}
