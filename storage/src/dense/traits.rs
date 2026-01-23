//! Dense trait implementations

use super::core::DenseStorage;
use crate::{AsAny, DataType, Result, Shape, Storage, StorageFormat, StorageFromVec, StorageToDense};
use alloc::vec::Vec;

impl<T: DataType> Storage<T> for DenseStorage<T> {
    fn format(&self) -> StorageFormat {
        StorageFormat::Dense
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn as_storage_ref(&self) -> &Self {
        self
    }

    fn full(dims: &[usize], value: T) -> Result<Self> {
        Self::full(dims, value)
    }

    fn map_structure<F>(&self, f: F) -> Result<Self>
    where
        F: FnMut(T) -> T,
    {
        let data = self.data.iter().cloned().map(f).collect();
        Ok(Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }
}

impl<T: DataType> StorageFromVec<T> for DenseStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self> {
        Self::from_vec(data, dims)
    }

    fn zeros(dims: &[usize]) -> Result<Self>
    where
        T: num_traits::Zero,
    {
        Self::zeros(dims)
    }

    fn ones(dims: &[usize]) -> Result<Self>
    where
        T: num_traits::One,
    {
        Self::ones(dims)
    }
}

impl<T: DataType> StorageToDense<T> for DenseStorage<T> {
    fn to_dense(&self) -> Result<DenseStorage<T>> {
        Ok(self.clone())
    }
}

impl<T: DataType + 'static> AsAny for DenseStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
