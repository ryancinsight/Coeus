//! CPU backend implementation using multi-threaded operations

use super::{Backend, BackendData, BackendError, Device, Result, Tensor, TensorData};
use coeus_dtype::Dtype;
use rayon::prelude::*;
use std::sync::Arc;

/// CPU backend for multi-threaded tensor operations
#[derive(Debug, Clone)]
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl<T: Dtype + num_traits::NumCast> Backend<T> for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    async fn allocate(&self, shape: &[usize]) -> Result<Arc<TensorData<T>>> {
        let numel = shape.iter().product();
        let data = vec![T::zero(); numel]; // Allocate with zeros
        Ok(Arc::new(TensorData {
            shape: shape.to_vec(),
            data: BackendData::Cpu(data),
        }))
    }

    async fn copy_from_host(&self, data: &[T], shape: &[usize]) -> Result<Tensor<T>> {
        let tensor_data = Arc::new(TensorData {
            shape: shape.to_vec(),
            data: BackendData::Cpu(data.to_vec()),
        });
        Ok(Tensor {
            data: tensor_data,
            shape: shape.to_vec(),
        })
    }

    async fn copy_to_host(&self, tensor: &Tensor<T>) -> Result<Vec<T>> {
        match &tensor.data.data {
            BackendData::Cpu(data) => Ok(data.clone()),
            BackendData::Gpu(_) => Err(BackendError::DeviceMismatch {
                required: Device::Cpu,
                actual: Device::Gpu,
            }),
        }
    }

    async fn ones(&self, shape: &[usize]) -> Result<Tensor<T>> {
        let numel = shape.iter().product();
        let data = vec![T::one(); numel];
        Ok(Tensor {
            data: Arc::new(TensorData {
                shape: shape.to_vec(),
                data: BackendData::Cpu(data),
            }),
            shape: shape.to_vec(),
        })
    }
    async fn add(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        Self::check_shapes_match(a, b)?;

        let result_data: Vec<T> = a
            .data
            .data
            .as_cpu()
            .iter()
            .zip(b.data.data.as_cpu().iter())
            .map(|(x, y)| *x + *y)
            .collect();

        self.copy_from_host(&result_data, &a.shape).await
    }

    async fn sub(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        Self::check_shapes_match(a, b)?;

        let result_data: Vec<T> = a
            .data
            .data
            .as_cpu()
            .iter()
            .zip(b.data.data.as_cpu().iter())
            .map(|(x, y)| *x - *y)
            .collect();

        self.copy_from_host(&result_data, &a.shape).await
    }

    async fn mul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        Self::check_shapes_match(a, b)?;

        let result_data: Vec<T> = a
            .data
            .data
            .as_cpu()
            .iter()
            .zip(b.data.data.as_cpu().iter())
            .map(|(x, y)| *x * *y)
            .collect();

        self.copy_from_host(&result_data, &a.shape).await
    }

    async fn div(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        Self::check_shapes_match(a, b)?;

        let result_data: Vec<T> = a
            .data
            .data
            .as_cpu()
            .iter()
            .zip(b.data.data.as_cpu().iter())
            .map(|(x, y)| *x / *y)
            .collect();

        self.copy_from_host(&result_data, &a.shape).await
    }

    async fn matmul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        Self::check_matmul_shapes(a, b)?;

        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        let mut result_data = vec![T::zero(); m * n];

        // Parallel matrix multiplication using Rayon
        result_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                let i = idx / n;
                let j = idx % n;

                let mut sum = T::zero();
                for l in 0..k {
                    let a_idx = i * k + l;
                    let b_idx = l * n + j;
                    sum = sum + a.data.data.as_cpu()[a_idx] * b.data.data.as_cpu()[b_idx];
                }
                *val = sum;
            });

        self.copy_from_host(&result_data, &[m, n]).await
    }

    async fn transpose(
        &self,
        _tensor: &Tensor<T>,
        _dim0: usize,
        _dim1: usize,
    ) -> Result<Tensor<T>> {
        Err(BackendError::invalid_operation(
            "Transpose not yet implemented",
        ))
    }

    async fn sum_dim(&self, tensor: &Tensor<T>, dim: usize) -> Result<Tensor<T>> {
        if dim >= tensor.shape().len() {
            return Err(BackendError::invalid_operation(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                tensor.shape().len()
            )));
        }

        let shape = tensor.shape();
        let data = tensor.data.data.as_cpu();

        // Calculate result shape (remove the summed dimension)
        let mut result_shape: Vec<usize> = shape.to_vec();
        result_shape.remove(dim);

        // If result is scalar, return sum of all elements
        if result_shape.is_empty() {
            let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
            return self.copy_from_host(&[sum], &[]).await;
        }

        // Calculate strides for the original tensor
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Calculate strides for the result tensor
        let mut result_strides = vec![1; result_shape.len()];
        for i in (0..result_shape.len() - 1).rev() {
            result_strides[i] = result_strides[i + 1] * result_shape[i + 1];
        }

        let result_size: usize = result_shape.iter().product();
        let mut result_data = vec![T::zero(); result_size];

        // Sum along the specified dimension using parallel processing
        result_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(result_idx, sum)| {
                // Convert result index to coordinates
                let mut coords = vec![0; result_shape.len()];
                let mut temp_idx = result_idx;
                for i in 0..result_shape.len() {
                    coords[i] = temp_idx / result_strides[i];
                    temp_idx %= result_strides[i];
                }

                // Insert the summed dimension coordinate (will be summed over)
                coords.insert(dim, 0);

                // Sum over all values along the specified dimension
                for i in 0..shape[dim] {
                    coords[dim] = i;

                    // Convert coordinates to linear index
                    let mut linear_idx = 0;
                    for j in 0..shape.len() {
                        linear_idx += coords[j] * strides[j];
                    }

                    *sum = *sum + data[linear_idx];
                }
            });

        self.copy_from_host(&result_data, &result_shape).await
    }

    async fn mean_dim(&self, tensor: &Tensor<T>, dim: usize) -> Result<Tensor<T>> {
        // Mean is sum divided by count along the dimension
        let sum_result = self.sum_dim(tensor, dim).await?;
        let count = tensor.shape()[dim];

        // Convert count to tensor element type
        let count_t = match num_traits::cast::<usize, T>(count) {
            Some(c) => c,
            None => {
                return Err(BackendError::invalid_operation(
                    "Cannot convert dimension size to tensor element type",
                ))
            }
        };

        let sum_data = sum_result.data.data.as_cpu();
        let result_data: Vec<T> = sum_data.iter().map(|&x| x / count_t).collect();

        self.copy_from_host(&result_data, sum_result.shape()).await
    }

    async fn cat(&self, tensors: &[&Tensor<T>], dim: usize) -> Result<Tensor<T>> {
        // Validate input
        if tensors.is_empty() {
            return Err(BackendError::invalid_operation(
                "Cannot concatenate empty tensor list",
            ));
        }

        if tensors.len() < 2 {
            return Err(BackendError::invalid_operation(
                "Need at least 2 tensors to concatenate",
            ));
        }

        // Validate shapes and dimension
        let first_shape = tensors[0].shape();
        if dim >= first_shape.len() {
            return Err(BackendError::invalid_operation(format!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                first_shape.len()
            )));
        }

        // Check that all tensors have compatible shapes for concatenation
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            let shape = tensor.shape();
            if shape.len() != first_shape.len() {
                return Err(BackendError::invalid_operation(format!(
                    "Tensor {} has {} dimensions, expected {}",
                    i,
                    shape.len(),
                    first_shape.len()
                )));
            }

            // Check dimensions other than the concatenation dimension
            for d in 0..shape.len() {
                if d != dim && shape[d] != first_shape[d] {
                    return Err(BackendError::invalid_operation(format!(
                        "Tensor {} has incompatible shape {:?} for concatenation along dimension {} with first tensor shape {:?}",
                        i, shape, dim, first_shape
                    )));
                }
            }
        }

        // Calculate output shape
        let mut output_shape = first_shape.to_vec();
        let mut total_size = 0usize;
        for tensor in tensors {
            total_size += tensor.shape()[dim];
        }
        output_shape[dim] = total_size;

        // Calculate total number of elements
        let total_elements: usize = output_shape.iter().product();

        // Create output data
        let mut output_data = Vec::with_capacity(total_elements);

        // Simple concatenation along the specified dimension
        // For now, implement a basic version that works for common cases
        if dim == tensors[0].shape().len() - 1 {
            // Concatenate along last dimension
            for tensor in tensors {
                if let BackendData::Cpu(data) = &tensor.data.data {
                    output_data.extend_from_slice(data);
                } else {
                    return Err(BackendError::invalid_operation("Expected CPU tensor data"));
                }
            }
        } else {
            // For other dimensions, this is more complex
            // For now, return an error indicating this case isn't implemented
            return Err(BackendError::invalid_operation(format!(
                "CPU concatenation along dimension {} not yet implemented",
                dim
            )));
        }

        // Create result tensor
        let result_data = Arc::new(TensorData {
            shape: output_shape.clone(),
            data: BackendData::Cpu(output_data),
        });

        Ok(Tensor {
            data: result_data,
            shape: output_shape,
        })
    }
}

impl CpuBackend {
    /// Check if two tensors have compatible shapes for element-wise operations
    fn check_shapes_match<T: Dtype>(a: &Tensor<T>, b: &Tensor<T>) -> Result<()> {
        if a.shape != b.shape {
            return Err(BackendError::ShapeMismatch {
                expected: a.shape.clone(),
                actual: b.shape.clone(),
            });
        }
        Ok(())
    }

    /// Check if shapes are compatible for matrix multiplication
    fn check_matmul_shapes<T: Dtype>(a: &Tensor<T>, b: &Tensor<T>) -> Result<()> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(BackendError::InvalidOperation {
                message: "Matrix multiplication requires 2D tensors".into(),
            });
        }

        if a.shape[1] != b.shape[0] {
            return Err(BackendError::ShapeMismatch {
                expected: vec![a.shape[1], b.shape[0]],
                actual: vec![a.shape[0], b.shape[1]],
            });
        }

        Ok(())
    }
}

/// Extension trait for BackendData to access CPU data
trait CpuDataAccess<T: Dtype> {
    fn as_cpu(&self) -> &[T];
}

impl<T: Dtype> CpuDataAccess<T> for BackendData<T> {
    fn as_cpu(&self) -> &[T] {
        match self {
            BackendData::Cpu(data) => data,
            BackendData::Gpu(_) => panic!("Cannot access GPU data on CPU backend"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cpu_tensor_operations() {
        let backend = CpuBackend::new();

        // Test addition
        let a = backend
            .copy_from_host(&[1.0, 2.0, 3.0], &[3])
            .await
            .unwrap();
        let b = backend
            .copy_from_host(&[4.0, 5.0, 6.0], &[3])
            .await
            .unwrap();
        let result = backend.add(&a, &b).await.unwrap();

        let result_data = backend.copy_to_host(&result).await.unwrap();
        assert_eq!(result_data, vec![5.0, 7.0, 9.0]);
    }

    #[tokio::test]
    async fn test_cpu_matrix_multiplication() -> crate::Result<()> {
        let backend = CpuBackend::new();

        // 2x3 matrix
        let a = backend
            .copy_from_host(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .await
            .unwrap();
        // 3x2 matrix
        let b = backend
            .copy_from_host(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])
            .await
            .unwrap();

        let result = backend.matmul(&a, &b).await?;
        let result_data = backend.copy_to_host(&result).await.unwrap();
        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        assert_eq!(result_data, vec![58.0, 64.0, 139.0, 154.0]);
        Ok(())
    }

    #[tokio::test]
    async fn test_cpu_shape_validation() {
        let backend = CpuBackend::new();

        let a = backend.copy_from_host(&[1.0, 2.0], &[2]).await.unwrap();
        let b = backend
            .copy_from_host(&[1.0, 2.0, 3.0], &[3])
            .await
            .unwrap();

        // Should fail due to shape mismatch
        let result = backend.add(&a, &b).await;
        assert!(result.is_err());
    }
}
