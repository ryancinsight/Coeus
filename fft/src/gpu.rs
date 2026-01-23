//! GPU-based FFT implementation using wgpu shaders

// TODO: GPU backend not yet implemented - disable GPU FFT for now
use coeus_error::Result;
use storage::DenseStorage;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuComplex32 {
    re: f32,
    im: f32,
}

/// GPU-based FFT processor (placeholder - not yet implemented)
pub struct GpuFft {
    size: usize,
}

impl GpuFft {
    /// Create a new GPU FFT processor (placeholder)
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    /// Perform forward FFT (placeholder - returns error)
    pub fn fft(
        &self,
        _input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        Err(coeus_error::BackendError::OperationNotSupported(
            "gpu_fft not implemented".to_string()
        ).into())
    }

    /// Perform inverse FFT (placeholder - returns error)
    pub fn ifft(
        &self,
        _input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        Err(coeus_error::BackendError::OperationNotSupported(
            "gpu_ifft not implemented".to_string()
        ).into())
    }

    /// Perform Real-to-Complex FFT (placeholder - returns error)
    pub fn rfft(
        &self,
        _input: &DenseStorage<dtype::float::Float32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        Err(coeus_error::BackendError::OperationNotSupported(
            "gpu_rfft not implemented".to_string()
        ).into())
    }

    /// Perform Complex-to-Real Inverse FFT (placeholder - returns error)
    pub fn irfft(
        &self,
        _input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::float::Float32>> {
        Err(coeus_error::BackendError::OperationNotSupported(
            "gpu_irfft not implemented".to_string()
        ).into())
    }

    /// Legacy forward compatibility (placeholder - returns error)
    pub fn forward(
        &self,
        _input: &DenseStorage<dtype::float::Float32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        Err(coeus_error::BackendError::OperationNotSupported(
            "gpu_forward not implemented".to_string()
        ).into())
    }

    /// Legacy inverse compatibility (placeholder - returns error)
    pub fn inverse(
        &self,
        _input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::float::Float32>> {
        Err(coeus_error::BackendError::OperationNotSupported(
            "gpu_inverse not implemented".to_string()
        ).into())
    }
}