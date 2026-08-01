use coeus_core::{Backend, BackendError};

mod convolution;
mod elementwise;
mod error;
mod impls;
mod matmul;
mod pool;
mod reduction;
mod unfold_fold;

/// CPU execution backend marker trait.
///
/// Implemented by [`MoiraiBackend`] and [`SequentialBackend`] from `coeus_core`.
/// The trait is sealed: external crates cannot add new implementations.
///
/// [`MoiraiBackend`]: coeus_core::MoiraiBackend
/// [`SequentialBackend`]: coeus_core::SequentialBackend
///
/// # Examples
///
/// ```
/// use coeus_ops::CpuBackend;
/// use coeus_core::SequentialBackend;
///
/// fn accept_cpu<B: CpuBackend>(_: &B) {}
///
/// let backend = SequentialBackend::new();
/// accept_cpu(&backend);
/// ```
///
/// The parent [`ComputeBackend`](coeus_core::ComputeBackend) trait is sealed,
/// so this marker remains restricted to first-party backends without a second
/// private sealing layer. CPU-addressable emulation backends can therefore use
/// the same canonical operation implementations instead of cloning them.
pub trait CpuBackend: Backend<Error = BackendError> {
    /// Borrow an `i64` device buffer as a mutable slice.
    fn as_mut_slice_i64<'a>(&self, buf: &'a mut Self::DeviceBuffer<i64>) -> &'a mut [i64];
}

impl CpuBackend for coeus_core::SequentialBackend {
    #[inline]
    fn as_mut_slice_i64<'a>(&self, buf: &'a mut Self::DeviceBuffer<i64>) -> &'a mut [i64] {
        use coeus_core::CpuAddressableStorageMut;
        buf.as_mut_slice()
    }
}

impl CpuBackend for coeus_core::MoiraiBackend {
    #[inline]
    fn as_mut_slice_i64<'a>(&self, buf: &'a mut Self::DeviceBuffer<i64>) -> &'a mut [i64] {
        use coeus_core::CpuAddressableStorageMut;
        buf.as_mut_slice()
    }
}
