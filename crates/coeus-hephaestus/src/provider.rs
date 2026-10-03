//! The provider seam every Hephaestus-backed Coeus backend is generic over.

use hephaestus_core::ComputeDevice;

/// Common provider identity and device acquisition seam.
///
/// # Safety
///
/// An implementation must ensure that every typed buffer exposed by its
/// [`hephaestus_core::ComputeDevice`] remains safe to retain behind an
/// `Arc` and to move between Coeus worker threads. Provider kernels must also
/// synchronize access according to their device API's contract.
pub unsafe trait HephaestusProvider: Send + Sync + Clone + Copy + Default + 'static {
    /// Concrete Hephaestus device type selected by this provider.
    type Device: ComputeDevice + Send + Sync + 'static;

    /// Stable backend name used by Coeus diagnostics.
    const NAME: &'static str;

    /// Return the lazily acquired device owned by this provider.
    fn device() -> &'static Self::Device;

    /// Try to acquire the provider device without panicking.
    ///
    /// # Errors
    ///
    /// Returns the provider's typed acquisition failure.
    fn try_device() -> hephaestus_core::Result<&'static Self::Device>;
}
