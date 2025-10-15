//! Complex number data types
//!
//! Implementation of complex number types using num-complex as backing.
//!
//! Reference: SRS-DTYPE-CPLX-001 (`Complex<f32>`), SRS-DTYPE-CPLX-002 (`Complex<f64>`)

#[cfg(feature = "complex")]
pub use num_complex::{Complex32, Complex64};

// When complex feature is not enabled, provide stub types for compilation
#[cfg(not(feature = "complex"))]
use core::marker::PhantomData;

/// Stub for 32-bit complex floating point type when complex feature is disabled
///
/// This type is only available when the `complex` feature is disabled.
/// Enable the `complex` feature to use actual complex number functionality.
#[cfg(not(feature = "complex"))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex32 {
    _phantom: PhantomData<()>,
}

/// Stub for 64-bit complex floating point type when complex feature is disabled
///
/// This type is only available when the `complex` feature is disabled.
/// Enable the `complex` feature to use actual complex number functionality.
#[cfg(not(feature = "complex"))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    _phantom: PhantomData<()>,
}

#[cfg(not(feature = "complex"))]
impl Complex32 {
    /// Creates a new stub complex number (no-op when complex feature disabled)
    #[must_use]
    pub fn new(_re: f32, _im: f32) -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Returns the norm of the complex number (always 0.0 for stub)
    #[must_use]
    pub fn norm(self) -> f32 {
        0.0
    }
}

#[cfg(not(feature = "complex"))]
impl Complex64 {
    /// Creates a new stub complex number (no-op when complex feature disabled)
    #[must_use]
    pub fn new(_re: f64, _im: f64) -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Returns the norm of the complex number (always 0.0 for stub)
    #[must_use]
    pub fn norm(self) -> f64 {
        0.0
    }
}
