//! Reduction operations for tensors
//!
//! This module contains operations that reduce tensor dimensions by computing
//! sums, means, and other aggregate operations across specified dimensions.

use crate::{Tensor, TensorError, Dtype, FloatDtype, Result};

impl<T: Dtype + num_traits::FromPrimitive + num_traits::ToPrimitive> Tensor<T> {
    /// Compute the sum of all elements
    ///
    /// # Returns
    /// A scalar tensor containing the sum of all elements
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let sum = tensor.sum();
    /// assert_eq!(sum.item().unwrap(), 10.0);
    /// ```
    pub fn sum(&self) -> Tensor<T> {
        let sum = self.data.iter().fold(T::zero(), |acc, x| acc + *x);
        Tensor::scalar(sum)
    }

    /// Compute the sum along specified dimensions
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to reduce. If None, reduces all dimensions
    /// * `keepdim` - Whether to keep reduced dimensions as size 1
    ///
    /// # Returns
    /// Result containing the reduced tensor or an error if reduction fails
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let sum_rows = tensor.sum_dim(Some(0), false).unwrap();
    /// // sum_rows: [4.0, 6.0] (sum along rows)
    /// ```
    pub fn sum_dim(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor<T>> {
        if let Some(d) = dim {
            if d >= self.shape.len() {
                return Err(TensorError::InvalidOperation {
                    message: format!("Dimension {} out of bounds for {}D tensor", d, self.shape.len())
                });
            }

            // For 2D tensors, implement dimension-wise reduction
            if self.shape.len() == 2 {
                if d == 0 {
                    // Sum along rows (reduce first dimension)
                    let cols = self.shape[1];
                    let mut result_data = vec![T::zero(); cols];
                    for c in 0..cols {
                        for r in 0..self.shape[0] {
                            result_data[c] = result_data[c] + self.data[r * cols + c];
                        }
                    }
                    let new_shape = if keepdim { vec![1, cols] } else { vec![cols] };
                    Ok(Tensor::from_vec(result_data, new_shape))
                } else if d == 1 {
                    // Sum along columns (reduce second dimension)
                    let rows = self.shape[0];
                    let mut result_data = vec![T::zero(); rows];
                    for r in 0..rows {
                        for c in 0..self.shape[1] {
                            result_data[r] = result_data[r] + self.data[r * self.shape[1] + c];
                        }
                    }
                    let new_shape = if keepdim { vec![rows, 1] } else { vec![rows] };
                    Ok(Tensor::from_vec(result_data, new_shape))
                } else {
                    Err(TensorError::InvalidOperation {
                        message: "Dimension reduction only implemented for 2D tensors".to_string()
                    })
                }
            } else {
                Err(TensorError::InvalidOperation {
                    message: "Dimension reduction only implemented for 2D tensors".to_string()
                })
            }
        } else {
            // Reduce all dimensions to scalar
            Ok(self.sum())
        }
    }

    /// Compute the mean of all elements
    ///
    /// # Returns
    /// A scalar tensor containing the mean of all elements
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let mean = tensor.mean();
    /// assert_eq!(mean.item().unwrap(), 2.5);
    /// ```
    pub fn mean(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let sum = self.data.iter().fold(T::zero(), |acc, x| acc + *x);
        let count = T::from_usize(self.data.len()).unwrap_or(T::one());
        Tensor::scalar(sum / count)
    }

    /// Compute the mean along specified dimensions
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to reduce. If None, reduces all dimensions
    /// * `keepdim` - Whether to keep reduced dimensions as size 1
    ///
    /// # Returns
    /// Result containing the reduced tensor or an error if reduction fails
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let mean_rows = tensor.mean_dim(Some(0), false).unwrap();
    /// // mean_rows: [2.0, 3.0] (mean along rows)
    /// ```
    pub fn mean_dim(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor<T>>
    where
        T: FloatDtype,
    {
        if let Some(d) = dim {
            if d >= self.shape.len() {
                return Err(TensorError::InvalidOperation {
                    message: format!("Dimension {} out of bounds for {}D tensor", d, self.shape.len())
                });
            }

            // For 2D tensors, implement dimension-wise reduction
            if self.shape.len() == 2 {
                let sum_tensor = self.sum_dim(Some(d), keepdim)?;
                let count = T::from_usize(if d == 0 { self.shape[0] } else { self.shape[1] })
                    .unwrap_or(T::one());

                let mean_data: Vec<T> = sum_tensor.data.iter().map(|x| *x / count).collect();
                Ok(Tensor::from_vec(mean_data, sum_tensor.shape))
            } else {
                Err(TensorError::InvalidOperation {
                    message: "Dimension reduction only implemented for 2D tensors".to_string()
                })
            }
        } else {
            // Reduce all dimensions to scalar
            Ok(self.mean())
        }
    }
}
