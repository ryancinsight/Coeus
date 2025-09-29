//! GPU Backend Stub
use super::{Backend, Tensor, TensorError};
use coeus_dtype::Dtype;
use thiserror::Error;

#[derive(Clone)]
pub struct GpuBackend;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("GPU operation not implemented")]
    Unsupported,
}

impl GpuBackend {
    pub fn new() -> Result<Self, GpuError> {
        // For now, always succeed but operations are stubs
        Ok(GpuBackend)
    }
}
    fn add(&self, _a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        Err(TensorError::Unsupported("GPU add stub".to_string()))
    }
    fn mul(&self, _a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        Err(TensorError::Unsupported("GPU mul stub".to_string()))
    }
    fn div(&self, _a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        Err(TensorError::Unsupported("GPU div stub".to_string()))
    }
    fn exp(&self, _a: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        Err(TensorError::Unsupported("GPU exp stub".to_string()))
    }
    fn create_tensor(&self, _data: Vec<T>, _shape: Vec<usize>) -> Result<Tensor<T>, TensorError> {
        Err(TensorError::Unsupported("GPU create stub".to_string()))
    }
    // Stub all other Backend methods similarly with Err
    fn sub(&self, _a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        Err(TensorError::Unsupported("GPU sub stub".to_string()))
    }
    fn neg(&self, _a: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        Err(TensorError::Unsupported("GPU neg stub".to_string()))
    }
    fn matmul(&self, _a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        Err(TensorError::Unsupported("GPU matmul stub".to_string()))
    }
    // ... continue for all required methods from traits.rs
}

