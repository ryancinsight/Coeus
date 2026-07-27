//! 3-D transposed-convolution capability seam.
//!
//! This trait is separate from [`super::ConvOps`] so a provider can add a
//! native 3-D transposed-convolution kernel without inheriting the CPU
//! implementation or a host-copy fallback.

use coeus_core::{ComputeBackend, Float, Layout, Scalar};

/// Backend-owned 3-D transposed-convolution dispatch.
///
/// CPU backends receive the canonical Leto-compatible scatter implementation.
/// Accelerator providers implement this trait only when they own a native
/// kernel for the complete operation contract.
pub trait ConvTranspose3dOps<T: Scalar>: ComputeBackend {
    /// Execute 3-D transposed convolution into the supplied output buffer.
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose3d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) where
        T: Float;
}
