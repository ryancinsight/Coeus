//! FFT (Fast Fourier Transform) operations for Coeus
//!
//! Provides high-performance FFT and IFFT operations with support for
//! multiple backends (CPU, GPU).

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;

pub use cpu::CpuFft;
