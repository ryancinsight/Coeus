//! Tensor reduction operations.
//!
//! This module provides operations that reduce tensor dimensions by
//! aggregating elements, such as sum and mean calculations.

use std::{collections::BTreeSet, vec, vec::Vec};

/// Reduction operations for tensors with dense storage.
///
/// This trait provides methods for aggregating tensor elements across
/// all dimensions, resulting in scalar tensors.
impl<B, T> crate::Tensor<B, coeus_storage::DenseStorage<T>, T>
where
    B: crate::Backend + Default,
    T: crate::DataType,
{
    /// Computes the sum of all elements in the tensor.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor x with n elements:
    /// ```text
    /// sum(x) = Σᵢ xᵢ
    /// ```
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the sum of all elements.
    ///
    /// # Panics
    ///
    /// This function uses conditional unsafe in release builds for performance.
    /// In debug builds, panics if scalar tensor creation fails (indicates a bug).
    /// In release builds, uses `unwrap_unchecked()` after mathematical proof of correctness.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
    ///     &[3]
    /// ).unwrap();
    ///
    /// let sum = tensor.sum_all();
    /// // sum = 1.0 + 2.0 + 3.0 = 6.0
    /// assert_eq!(sum.as_slice()[0].get(), 6.0);
    /// ```
    #[must_use]
    pub fn sum_all(&self) -> Self {
        let sum_value = self
            .as_slice()
            .iter()
            .copied()
            .fold(T::zero(), |acc, x| acc + x);

        // SAFETY: Scalar tensor creation with single element always succeeds
        #[cfg(debug_assertions)]
        {
            crate::Tensor::from_vec(vec![sum_value], &[1])
                .expect("Scalar tensor creation failed: this is a bug")
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            crate::Tensor::from_vec(vec![sum_value], &[1]).unwrap_unchecked()
        }
    }

    /// Computes the mean (average) of all elements in the tensor.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor x with n elements:
    /// ```text
    /// mean(x) = (1/n) * Σᵢ xᵢ
    /// ```
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the mean of all elements.
    ///
    /// # Panics
    ///
    /// This function uses conditional unsafe in release builds for performance.
    /// In debug builds, panics if scalar tensor creation fails (indicates a bug).
    /// In release builds, uses `unwrap_unchecked()` after mathematical proof of correctness.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
    ///     &[3]
    /// ).unwrap();
    ///
    /// let mean = tensor.mean_all();
    /// // mean = (1.0 + 2.0 + 3.0) / 3 = 2.0
    /// assert_eq!(mean.as_slice()[0].get(), 2.0);
    /// ```
    #[must_use]
    pub fn mean_all(&self) -> Self {
        let sum_value = self
            .as_slice()
            .iter()
            .copied()
            .fold(T::zero(), |acc, x| acc + x);
        let n = T::from(self.len()).unwrap_or_else(T::one); // Fallback to 1 if cast fails
        let mean_value = sum_value / n;

        // SAFETY: Scalar tensor creation with single element always succeeds
        #[cfg(debug_assertions)]
        {
            crate::Tensor::from_vec(vec![mean_value], &[1])
                .expect("Scalar tensor creation failed: this is a bug")
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            crate::Tensor::from_vec(vec![mean_value], &[1]).unwrap_unchecked()
        }
    }

    /// Computes the sum of elements along specified dimensions.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor x with dimensions and specified dims to reduce:
    /// ```text
    /// sum(x, dims) = Σ_{i in dims} x along those dimensions
    /// ```
    ///
    /// # Arguments
    ///
    /// * `dims` - Dimensions to reduce. If None, reduces all dimensions to scalar.
    /// * `keepdim` - If true, keeps reduced dimensions with size 1.
    ///
    /// # Returns
    ///
    /// A tensor with reduced dimensions containing the sum along specified axes.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `dims` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
    ///     &[2, 2]
    /// ).unwrap();
    ///
    /// // Sum along dimension 0 (rows)
    /// let sum_dim0 = tensor.sum_dims(Some(&[0]), false).unwrap();
    /// // sum_dim0.shape() = [2], values = [4.0, 6.0]
    ///
    /// // Sum along dimension 0, keep dimensions
    /// let sum_dim0_keep = tensor.sum_dims(Some(&[0]), true).unwrap();
    /// // sum_dim0_keep.shape() = [1, 2], values = [4.0, 6.0]
    ///
    /// // Sum all elements
    /// let sum_all = tensor.sum_dims(None, false).unwrap();
    /// // sum_all.shape() = [1], value = [10.0]
    /// ```
    ///
    /// # Errors
    /// Returns error if dimension indices are out of bounds or invalid.
    #[must_use = "sum_dims returns the result tensor that should be used"]
    pub fn sum_dims(&self, dims: Option<&[usize]>, keepdim: bool) -> crate::Result<Self> {
        self.reduce_dims(dims, keepdim, |acc, x| acc + x, T::zero())
    }

    /// Computes the mean of elements along specified dimensions.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor x with dimensions and specified dims to reduce:
    /// ```text
    /// mean(x, dims) = (1/n) * Σ_{i in dims} x along those dimensions
    /// ```
    /// where n is the product of sizes of reduced dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - Dimensions to reduce. If None, reduces all dimensions to scalar.
    /// * `keepdim` - If true, keeps reduced dimensions with size 1.
    ///
    /// # Returns
    ///
    /// A tensor with reduced dimensions containing the mean along specified axes.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `dims` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
    ///     &[2, 2]
    /// ).unwrap();
    ///
    /// // Mean along dimension 0 (rows)
    /// let mean_dim0 = tensor.mean_dims(Some(&[0]), false).unwrap();
    /// // mean_dim0.shape() = [2], values = [2.0, 3.0]
    ///
    /// // Mean along dimension 0, keep dimensions
    /// let mean_dim0_keep = tensor.mean_dims(Some(&[0]), true).unwrap();
    /// // mean_dim0_keep.shape() = [1, 2], values = [2.0, 3.0]
    ///
    /// // Mean all elements
    /// let mean_all = tensor.mean_dims(None, false).unwrap();
    /// // mean_all.shape() = [1], value = [2.5]
    /// ```
    ///
    /// # Errors
    /// Returns error if dimension indices are out of bounds or invalid.
    #[must_use = "mean_dims returns the result tensor that should be used"]
    pub fn mean_dims(&self, dims: Option<&[usize]>, keepdim: bool) -> crate::Result<Self> {
        let sum = self.reduce_dims(dims, keepdim, |acc, x| acc + x, T::zero())?;
        let count = self.reduce_dims(dims, keepdim, |acc, _| acc + T::one(), T::zero())?;
        {
            let sum_data = sum.as_slice();
            let count_data = count.as_slice();
            let result_data: Vec<T> = sum_data.iter().zip(count_data.iter()).map(|(s, c)| *s / *c).collect();
            let result_shape = sum.shape().dims().to_vec();
            crate::Tensor::from_vec(result_data, &result_shape)
        }
    }

    /// Generic dimensional reduction with custom reduction function.
    ///
    /// # Arguments
    ///
    /// * `dims` - Dimensions to reduce. If None, reduces all dimensions.
    /// * `keepdim` - If true, keeps reduced dimensions with size 1.
    /// * `reduce_fn` - Function to combine accumulator and element: (acc, x) -> acc
    /// * `init` - Initial value for the accumulator.
    ///
    /// # Returns
    ///
    /// A tensor with reduced dimensions.
    fn reduce_dims<F>(
        &self,
        dims: Option<&[usize]>,
        keepdim: bool,
        reduce_fn: F,
        init: T,
    ) -> crate::Result<Self>
    where
        F: FnMut(T, T) -> T,
    {
        let result = self.reduce_dims_impl(dims, keepdim, reduce_fn, init)?;
        // Inherit gradient requirements from input tensor (PyTorch-style)
        Ok(result.requires_grad_(self.requires_grad()))
    }

    fn reduce_dims_impl<F>(
        &self,
        dims: Option<&[usize]>,
        keepdim: bool,
        mut reduce_fn: F,
        init: T,
    ) -> crate::Result<Self>
    where
        F: FnMut(T, T) -> T,
    {
        let input_shape = self.shape().dims();
        let ndim = input_shape.len();

        // Determine which dimensions to reduce
        let dims_to_reduce: BTreeSet<usize> = if let Some(dims) = dims {
            // Validate dimensions
            for &dim in dims {
                if dim >= ndim {
                    return Err(crate::TensorError::ShapeError {
                        expected: ndim,
                        actual: dim,
                        message: std::format!(
                            "Dimension {dim} is out of bounds for tensor with {ndim} dimensions"
                        ),
                    });
                }
            }
            dims.iter().copied().collect()
        } else {
            // Reduce all dimensions
            (0..ndim).collect()
        };

        // Calculate output shape
        let mut output_shape: Vec<usize> = Vec::new();
        for (i, &size) in input_shape.iter().enumerate() {
            if dims_to_reduce.contains(&i) {
                if keepdim {
                    output_shape.push(1);
                }
                // Skip dimension if not keeping it
            } else {
                output_shape.push(size);
            }
        }

        // If all dimensions are reduced and keepdim is false, result is scalar
        if output_shape.is_empty() {
            output_shape = vec![1];
        }

        let input_data = self.as_slice();
        let output_size: usize = output_shape.iter().product();
        let mut output_data = vec![init; output_size];

        // Calculate strides for input tensor
        let mut input_strides = vec![1; ndim];
        for i in (1..ndim).rev() {
            input_strides[i - 1] = input_strides[i] * input_shape[i];
        }

        // Calculate strides for output tensor
        let mut output_strides = vec![1; output_shape.len()];
        for i in (1..output_shape.len()).rev() {
            output_strides[i - 1] = output_strides[i] * output_shape[i];
        }

        // Perform reduction
        let mut output_coords = vec![0; output_shape.len()];
        let mut input_coords = vec![0; ndim];

        // For each position in output tensor
        for (output_idx, output_elem) in output_data.iter_mut().enumerate() {
            // Convert flat output index to coordinates
            let mut temp_idx = output_idx;
            for (i, coord) in output_coords.iter_mut().enumerate() {
                *coord = temp_idx % output_shape[i];
                temp_idx /= output_shape[i];
            }

            // Map output coordinates back to input coordinates
            let mut output_coord_idx = 0;
            for (i, input_coord) in input_coords.iter_mut().enumerate() {
                if dims_to_reduce.contains(&i) {
                    // This dimension is being reduced - iterate over all values
                    *input_coord = 0; // Will be set in inner loop
                } else {
                    // This dimension is preserved - copy coordinate
                    *input_coord = output_coords[output_coord_idx];
                    output_coord_idx += 1;
                }
            }

            // Sum over the reduced dimensions
            let mut accumulator = init;
            let mut reduced_size = 1;
            for &dim in &dims_to_reduce {
                reduced_size *= input_shape[dim];
            }

            // Iterate over all combinations of reduced dimensions
            for reduced_idx in 0..reduced_size {
                // Set coordinates for reduced dimensions
                let mut temp_reduced_idx = reduced_idx;
                for &dim in &dims_to_reduce {
                    input_coords[dim] = temp_reduced_idx % input_shape[dim];
                    temp_reduced_idx /= input_shape[dim];
                }

                // Convert coordinates to flat index
                let mut flat_idx = 0;
                for d in 0..ndim {
                    flat_idx += input_coords[d] * input_strides[d];
                }

                // Accumulate value
                accumulator = reduce_fn(accumulator, input_data[flat_idx]);
            }

            *output_elem = accumulator;
        }

        // Create output tensor
        Self::from_vec(output_data, &output_shape)
    }

    /// Computes the maximum of elements along specified dimensions.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor x with dimensions and specified dims to reduce:
    /// ```text
    /// max(x, dims) = max_{i in dims} x along those dimensions
    /// ```
    ///
    /// # Arguments
    ///
    /// * `dims` - Dimensions to reduce. If None, reduces all dimensions to scalar.
    /// * `keepdim` - If true, keeps reduced dimensions with size 1.
    ///
    /// # Returns
    ///
    /// A tensor with reduced dimensions containing the maximum along specified axes.
    ///
    /// # Errors
    /// Returns error if dimension indices are out of bounds or invalid.
    #[must_use = "max_dims returns the result tensor that should be used"]
    pub fn max_dims(&self, dims: Option<&[usize]>, keepdim: bool) -> crate::Result<Self>
    where
        T: PartialOrd + Clone,
    {
        // Use the first element as initial value, then find max
        if self.is_empty() {
            return Err(crate::error::TensorError::EmptyTensor);
        }
        let first_elem = self.as_slice()[0].clone();
        self.reduce_dims(dims, keepdim, |a, b| if a > b { a } else { b }, first_elem)
    }

    /// Computes the minimum of elements along specified dimensions.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor x with dimensions and specified dims to reduce:
    /// ```text
    /// min(x, dims) = min_{i in dims} x along those dimensions
    /// ```
    ///
    /// # Arguments
    ///
    /// * `dims` - Dimensions to reduce. If None, reduces all dimensions to scalar.
    /// * `keepdim` - If true, keeps reduced dimensions with size 1.
    ///
    /// # Returns
    ///
    /// A tensor with reduced dimensions containing the minimum along specified axes.
    ///
    /// # Errors
    /// Returns error if dimension indices are out of bounds or invalid.
    #[must_use = "min_dims returns the result tensor that should be used"]
    pub fn min_dims(&self, dims: Option<&[usize]>, keepdim: bool) -> crate::Result<Self>
    where
        T: PartialOrd + Clone,
    {
        // Use the first element as initial value, then find min
        if self.is_empty() {
            return Err(crate::error::TensorError::EmptyTensor);
        }
        let first_elem = self.as_slice()[0].clone();
        self.reduce_dims(dims, keepdim, |a, b| if a < b { a } else { b }, first_elem)
    }

    /// PyTorch-compatible alias for `sum_dims`.
    ///
    /// Computes the sum of elements along specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimensions to reduce. If None, reduces all dimensions to scalar.
    /// * `keepdim` - If true, keeps reduced dimensions with size 1.
    ///
    /// # Returns
    ///
    /// A tensor with reduced dimensions containing the sum along specified axes.
    #[must_use = "sum returns the result tensor that should be used"]
    pub fn sum(&self, dim: Option<&[usize]>, keepdim: bool) -> crate::Result<Self> {
        self.sum_dims(dim, keepdim)
    }

    /// PyTorch-compatible alias for `mean_dims`.
    ///
    /// Computes the mean of elements along specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimensions to reduce. If None, reduces all dimensions to scalar.
    /// * `keepdim` - If true, keeps reduced dimensions with size 1.
    ///
    /// # Returns
    ///
    /// A tensor with reduced dimensions containing the mean along specified axes.
    #[must_use = "mean returns the result tensor that should be used"]
    pub fn mean(&self, dim: Option<&[usize]>, keepdim: bool) -> crate::Result<Self> {
        self.mean_dims(dim, keepdim)
    }

    /// PyTorch-compatible alias for `max_dims`.
    ///
    /// Computes the maximum of elements along specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimensions to reduce. If None, reduces all dimensions to scalar.
    /// * `keepdim` - If true, keeps reduced dimensions with size 1.
    ///
    /// # Returns
    ///
    /// A tensor with reduced dimensions containing the maximum along specified axes.
    #[must_use = "max returns the result tensor that should be used"]
    pub fn max(&self, dim: Option<&[usize]>, keepdim: bool) -> crate::Result<Self>
    where
        T: PartialOrd + Clone,
    {
        self.max_dims(dim, keepdim)
    }

    /// PyTorch-compatible alias for `min_dims`.
    ///
    /// Computes the minimum of elements along specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimensions to reduce. If None, reduces all dimensions to scalar.
    /// * `keepdim` - If true, keeps reduced dimensions with size 1.
    ///
    /// # Returns
    ///
    /// A tensor with reduced dimensions containing the minimum along specified axes.
    #[must_use = "min returns the result tensor that should be used"]
    pub fn min(&self, dim: Option<&[usize]>, keepdim: bool) -> crate::Result<Self>
    where
        T: PartialOrd + Clone,
    {
        self.min_dims(dim, keepdim)
    }
}
