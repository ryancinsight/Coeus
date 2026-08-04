use coeus_core::{ComputeBackend, Layout, Scalar};

/// Backend-selected half-vector rotation used by rotary embeddings.
pub trait RotateHalfOps<T: Scalar>: ComputeBackend {
    /// Allocate and initialize storage for `[-x₂, x₁]` along the final axis.
    ///
    /// # Errors
    ///
    /// Returns a typed backend failure for an invalid rank, odd final extent,
    /// allocation failure, layout failure, or provider dispatch failure.
    fn rotate_half_storage(
        &self,
        input: &Self::DeviceBuffer<T>,
        layout: &Layout,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error>;
}
