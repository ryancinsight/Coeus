//! COO trait implementations

use super::core::CooStorage;
use crate::{AsAny, DataType, Result, Shape, Storage, StorageError, StorageFormat};
use alloc::vec::Vec;
use core::fmt;

impl<T: DataType> Storage<T> for CooStorage<T> {
    fn map_structure<F>(&self, f: F) -> Result<Self>
    where
        Self: Sized,
        F: FnMut(T) -> T,
    {
        let data = self.data.iter().cloned().map(f).collect();
        Ok(Self {
            data,
            row_indices: self.row_indices.clone(),
            col_indices: self.col_indices.clone(),
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
        StorageFormat::Coo
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
                reason: "COO storage requires 2D shape",
            });
        }

        let rows = dims[0];
        let cols = dims[1];

        if value == T::zero() {
            Self::empty(dims)
        } else {
            let total = rows * cols;
            let mut data = Vec::with_capacity(total);
            let mut row_indices = Vec::with_capacity(total);
            let mut col_indices = Vec::with_capacity(total);

            for row in 0..rows {
                for col in 0..cols {
                    data.push(value);
                    row_indices.push(row);
                    col_indices.push(col);
                }
            }

            Self::new(data, row_indices, col_indices, dims)
        }
    }
}

impl<T: DataType> crate::StorageToDense<T> for CooStorage<T>
where
    T: num_traits::Zero,
{
    fn to_dense(&self) -> Result<crate::DenseStorage<T>> {
        self.to_dense()
    }
}

impl<T: DataType + 'static> AsAny for CooStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: DataType + fmt::Display + num_traits::Zero> fmt::Display for CooStorage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.dims();
        writeln!(f, "CooStorage {}x{} (nnz={})", rows, cols, self.nnz())?;

        for i in 0..self.nnz().min(20) {
            writeln!(
                f,
                "  ({}, {}): {}",
                self.row_indices[i], self.col_indices[i], self.data[i]
            )?;
        }
        if self.nnz() > 20 {
            writeln!(f, "  ... ({} more entries)", self.nnz() - 20)?;
        }

        Ok(())
    }
}

impl<T: DataType> crate::StorageFromVec<T> for CooStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self>
    where
        Self: Sized,
    {
        use alloc::vec::Vec as AllocVec;
        
        if dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "COO storage requires 2D shape",
            });
        }
        
        let _rows = dims[0];
        let cols = dims[1];
        
        // For COO from dense vec: only store non-zeros
        let mut nz_data = AllocVec::new();
        let mut row_indices = AllocVec::new();
        let mut col_indices = AllocVec::new();
        
        for (idx, &val) in data.iter().enumerate() {
            if val != T::zero() {
                let row = idx / cols;
                let col = idx % cols;
                nz_data.push(val);
                row_indices.push(row);
                col_indices.push(col);
            }
        }
        
        Self::new(nz_data, row_indices, col_indices, dims)
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
