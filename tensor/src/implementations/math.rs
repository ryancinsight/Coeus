//! Mathematical methods for Tensor


use crate::{Backend, DataType, Result, Storage, Tensor};
use storage::StorageFromVec;

impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Checks if the tensor contains any NaN values.
    pub fn is_nan(&self) -> bool
    where
        T: num_traits::Float,
    {
        // Check for NaN values
        self.as_slice().iter().any(|&x| x.is_nan())
    }

    /// Checks if the tensor contains any infinite values.
    pub fn is_inf(&self) -> bool
    where
        T: num_traits::Float,
    {
        // Check for infinite values using proper float methods
        self.as_slice().iter().any(|&x| x.is_infinite())
    }

    /// Clamps tensor values to a specified range in place.
    pub fn clamp_(&mut self, min: T, max: T) -> Result<()>
    where
        T: PartialOrd + Copy,
        S: Storage<T>,
    {
        let data = self.storage.as_mut_slice();
        for x in data {
            if *x < min {
                *x = min;
            } else if *x > max {
                *x = max;
            }
        }
        Ok(())
    }

    /// Clamps tensor values to a specified range, returning a new tensor.
    pub fn clamp(&self, min: T, max: T) -> Result<Tensor<B, S, T>>
    where
        T: PartialOrd + Copy,
        S: Storage<T> + Clone,
        B: Backend<Data = T>,
    {
        let mut result = self.clone();
        result.clamp_(min, max)?;
        Ok(result)
    }

    /// Clamps tensor values to a minimum value in place.
    pub fn clamp_min_(&mut self, min: T) -> Result<()>
    where
        T: PartialOrd + Copy,
        S: Storage<T>,
    {
        let data = self.storage.as_mut_slice();
        for x in data {
            if *x < min {
                *x = min;
            }
        }
        Ok(())
    }

    /// Clamps tensor values to a minimum value, returning a new tensor.
    pub fn clamp_min(&self, min: T) -> Result<Tensor<B, S, T>>
    where
        T: PartialOrd + Copy,
        S: Storage<T> + Clone,
        B: Backend<Data = T>,
    {
        let mut result = self.clone();
        result.clamp_min_(min)?;
        Ok(result)
    }

    /// Clamps tensor values to a maximum value in place.
    pub fn clamp_max_(&mut self, max: T) -> Result<()>
    where
        T: PartialOrd + Copy,
        S: Storage<T>,
    {
        let data = self.storage.as_mut_slice();
        for x in data {
            if *x > max {
                *x = max;
            }
        }
        Ok(())
    }

    /// Clamps tensor values to a maximum value, returning a new tensor.
    pub fn clamp_max(&self, max: T) -> Result<Tensor<B, S, T>>
    where
        T: PartialOrd + Copy,
        S: Storage<T> + Clone,
        B: Backend<Data = T>,
    {
        let mut result = self.clone();
        result.clamp_max_(max)?;
        Ok(result)
    }


    /// Zeros all elements of this tensor in-place.
    pub fn zero_(&mut self)
    where
        T: Default + Copy,
    {
        let data = self.as_mut_slice();
        let zero = T::default();
        for elem in data.iter_mut() {
            *elem = zero;
        }
    }
}
