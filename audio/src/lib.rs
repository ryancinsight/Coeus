//! # Coeus Audio Processing Library
//!
//! High-performance audio processing operations with GPU acceleration support.
//!
//! ## Features
//!
//! - **Fast Fourier Transform (FFT)**: High-performance FFT/IFFT with both CPU and GPU implementations
//! - **GPU Acceleration**: Leverages WebGPU for parallel audio processing on modern GPUs
//! - **Real-time Processing**: Optimized for low-latency audio applications
//! - **Flexible API**: Easy integration with tensor operations and ML frameworks
//!
//! ## Quick Start
//!
//! ```rust
//! use audio::{Fft, GpuFft};
//!
//! // CPU FFT processing
//! let mut fft = Fft::new(1024).unwrap();
//!
//! // GPU FFT processing (async)
//! #[cfg(feature = "gpu")]
//! {
//!     let backend = std::sync::Arc::new(backend::gpu::GpuBackend::new().await.unwrap());
//!     let gpu_fft = GpuFft::new(backend, 1024).unwrap();
//!
//!     let input = vec![0.0; 1024]; // Your audio samples
//!     let result = gpu_fft.forward_real(&input).await.unwrap();
//! }
//! ```

pub mod error;
pub mod fft;

pub use error::{AudioError, AudioResult};
pub use fft::{Fft, GpuFft};
