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
//! use audio::Fft;
//! #[cfg(feature = "gpu")]
//! use audio::GpuFft;
//!
//! // CPU FFT processing
//! let mut fft = Fft::new(1024).unwrap();
//!
//! // GPU FFT processing (async)
//! #[cfg(feature = "gpu")]
//! {
//!     let backend = std::sync::Arc::new(
//!         futures::executor::block_on(backend::gpu::GpuBackend::new()).unwrap(),
//!     );
//!     let gpu_fft = GpuFft::new(backend, 1024).unwrap();
//!
//!     let input = vec![0.0; 1024]; // Your audio samples
//!     let _result = futures::executor::block_on(gpu_fft.forward_real(&input)).unwrap();
//! }
//! ```

pub mod error;
pub mod fft;

pub use error::{AudioError, AudioResult};
pub use fft::Fft;
#[cfg(feature = "gpu")]
pub use fft::GpuFft;
