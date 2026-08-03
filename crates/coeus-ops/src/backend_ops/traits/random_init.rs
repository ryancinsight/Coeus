//! Seeded random-initialization dispatch.

use coeus_core::{ComputeBackend, Layout, Scalar};

/// Backend-selected seeded random storage initialization.
///
/// Implementations allocate and initialize one replacement buffer through the
/// provider selected by the backend type. Selection occurs once per tensor;
/// inner generation remains monomorphized by scalar and rank.
pub trait RandomInitOps<T: Scalar>: ComputeBackend {
    /// Allocate storage filled with uniform samples in `[low, high)`.
    ///
    /// # Errors
    ///
    /// Returns a typed backend failure when rank, allocation, generation, or
    /// transfer fails.
    fn uniform_random(
        &self,
        layout: &Layout,
        low: T,
        high: T,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error>;

    /// Allocate storage filled with normal samples.
    ///
    /// # Errors
    ///
    /// Returns a typed backend failure when rank, allocation, generation, or
    /// transfer fails.
    fn normal_random(
        &self,
        layout: &Layout,
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error>;
}
