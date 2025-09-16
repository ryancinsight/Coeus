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
impl<T: Dtype> Backend<T> for CpuBackend {
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

    async fn sum_dim(&self, _tensor: &Tensor<T>, _dim: usize) -> Result<Tensor<T>> {
        Err(BackendError::invalid_operation(
            "Sum dim not yet implemented",
        ))
    }

    async fn mean_dim(&self, _tensor: &Tensor<T>, _dim: usize) -> Result<Tensor<T>> {
        Err(BackendError::invalid_operation(
            "Mean dim not yet implemented",
        ))
    }

    async fn cat(&self, _tensors: &[&Tensor<T>], _dim: usize) -> Result<Tensor<T>> {
        Err(BackendError::invalid_operation("Cat not yet implemented"))
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
