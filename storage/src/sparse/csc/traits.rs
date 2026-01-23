//! CSC trait implementations

use super::core::CscStorage;
use crate::{AsAny, DataType, Result, Shape, Storage, StorageError, StorageFormat};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

impl<T: DataType> Storage<T> for CscStorage<T> {
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
        StorageFormat::Csc
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
                reason: "CSC storage requires 2D shape",
            });
        }

        let rows = dims[0];
        let cols = dims[1];

        if value == T::zero() {
            Self::empty(dims)
        } else {
            let total = rows * cols;
            let mut data = alloc::vec::Vec::with_capacity(total);
            let mut indices = alloc::vec::Vec::with_capacity(total);
            let mut indptr = vec![0];

            for _col in 0..cols {
                for row in 0..rows {
                    data.push(value);
                    indices.push(row);
                }
                indptr.push(data.len());
            }

            Self::new(data, indices, indptr, dims)
        }
    }
}

impl<T: DataType> crate::StorageToDense<T> for CscStorage<T>
where
    T: num_traits::Zero,
{
    fn to_dense(&self) -> Result<crate::DenseStorage<T>> {
        self.to_dense()
    }
}

impl<T: DataType + 'static> AsAny for CscStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: DataType + fmt::Display + num_traits::Zero> fmt::Display for CscStorage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.dims();
        writeln!(f, "CscStorage {}x{} (nnz={})", rows, cols, self.nnz())
    }
}

impl<T: DataType> crate::StorageFromVec<T> for CscStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self>
    where
        Self: Sized,
    {
        use alloc::vec::Vec;
        
        if dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "CSC storage requires 2D shape",
            });
        }
        
        let rows = dims[0];
        let cols = dims[1];
        
        // Convert row-major dense to CSC (column-major sparse)
        let mut nz_data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];
        
        for col in 0..cols {
            for row in 0..rows {
                let val = data[row * cols + col];
                if val != T::zero() {
                    nz_data.push(val);
                    indices.push(row);
                }
            }
            indptr.push(nz_data.len());
        }
        
        Self::new(nz_data, indices, indptr, dims)
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
