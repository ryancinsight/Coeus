//! CUDA backend stub (forward to cuBLAS).

use super::Backend;
use std::env;
use cuda_sys::{cublasCreate, cublasSgemm, cublasHandle_t};

pub struct CudaBackend {
    handle: cublasHandle_t,
}

impl CudaBackend {
    pub fn new() -> Result<Self, BackendError> {
        let path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        unsafe {
            let mut handle: cublasHandle_t = std::ptr::null_mut();
            cublasCreate(&mut handle);
            Ok(Self { handle })
        }
    }
}

impl Backend for CudaBackend {
    type Dtype = f32;
    type TensorData = Vec<Self::Dtype>; // Stub: cuBLAS matrices

    fn add(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData {
        // Forward to cublasSaxpy or similar
        todo!("cuBLAS add impl")
    }

    fn matmul(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData {
        // cuBLAS sgemm
        let m = /* from shape */ 1;
        let n = /* from shape */ 1;
        let k = /* from shape */ 1;
        unsafe {
            cublasSgemm(self.handle, /* params */);
        }
        vec![0.0; m * n] // Placeholder
    }
}
