//! Scaled dot-product attention sub-trait.
//!
//! [`AttentionOps`] is the interface-segregated sub-trait for SDPA
//! forward and backward kernel dispatch.

use coeus_core::{ComputeBackend, Float, Layout, Scalar};

/// Scalar types supported by the provider-owned attention contract.
///
/// The marker preserves [`BackendOps`](super::super::BackendOps) for scalar
/// types that do not define attention while making provider support explicit
/// at each attention call.
pub trait AttentionScalar: Float {}

impl AttentionScalar for f32 {}
impl AttentionScalar for f64 {}

/// Scaled dot-product attention operations.
///
/// Backends implement this optional capability independently from the
/// universal [`BackendOps`](super::super::BackendOps) surface. This keeps
/// non-floating scalar kernels available on devices whose attention provider
/// supports a narrower scalar set.
pub trait AttentionOps<T: Scalar>: ComputeBackend {
    /// Scaled dot-product attention forward.
    ///
    /// # Errors
    ///
    /// Returns a typed backend failure. Validation and preparation failures
    /// occur before caller-visible mutation; dispatch failures may leave
    /// destination contents unspecified.
    fn sdp_attention(
        &self,
        query: &Self::DeviceBuffer<T>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<T>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<T>,
        value_layout: &Layout,
        key_padding_mask: Option<&Self::DeviceBuffer<T>>,
        key_padding_mask_layout: Option<&Layout>,
        is_causal: bool,
        scale: T,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
        attn_weights: &mut Self::DeviceBuffer<T>,
        attn_weights_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: AttentionScalar;

    /// Scaled dot-product attention backward.
    ///
    /// # Errors
    ///
    /// Returns a typed backend failure. Validation and preparation failures
    /// occur before accumulation; dispatch failures may partially modify the
    /// selected destinations.
    #[expect(
        clippy::too_many_arguments,
        reason = "the method carries the complete differentiable attention contract"
    )]
    fn sdp_attention_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        query: &Self::DeviceBuffer<T>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<T>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<T>,
        value_layout: &Layout,
        attn_weights: &Self::DeviceBuffer<T>,
        attn_weights_layout: &Layout,
        scale: T,
        grad_q: Option<(&mut Self::DeviceBuffer<T>, &Layout)>,
        grad_k: Option<(&mut Self::DeviceBuffer<T>, &Layout)>,
        grad_v: Option<(&mut Self::DeviceBuffer<T>, &Layout)>,
    ) -> Result<(), Self::Error>
    where
        T: AttentionScalar;
}
