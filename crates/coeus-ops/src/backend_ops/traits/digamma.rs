//! Optional provider capability for real digamma. Device providers opt in.
use coeus_core::{ComputeBackend, Float, Layout};

/// Real digamma kernel, used by gamma-log reverse-mode differentiation.
///
/// This capability is independent of the exhaustive unary dispatch vocabulary.
/// A backend without a provider implementation cannot instantiate the tracked
/// gamma-log operation; the API never transfers an unsupported device to CPU.
///
/// # Safety
///
/// On success every logical output element must be initialized without reading
/// its previous contents. Callers discard uninitialized output after an error.
pub unsafe trait DigammaOps<T: Float>: ComputeBackend {
    /// Write `digamma(input)` under the provider scalar contract over the layouts.
    fn elementwise_digamma(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;
}
