//! Scaled dot-product attention sub-trait.
//!
//! [`AttentionOps`] is the interface-segregated sub-trait for SDPA
//! forward and backward kernel dispatch.

use coeus_core::{ComputeBackend, Float, Layout, Scalar};

/// Scaled dot-product attention operations.
///
/// This sub-trait is one of seven concerns that compose
/// [`BackendOps`].  Backends implement `AttentionOps` directly; the
/// blanket impl provides `BackendOps` automatically.
///
/// [`BackendOps`]: super::super::BackendOps
pub trait AttentionOps<T: Scalar>: ComputeBackend {
    /// Scaled dot-product attention forward.
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
    ) where
        T: Float;

    /// Scaled dot-product attention backward.
    #[allow(clippy::too_many_arguments)]
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
        grad_q: Option<&mut Self::DeviceBuffer<T>>,
        grad_k: Option<&mut Self::DeviceBuffer<T>>,
        grad_v: Option<&mut Self::DeviceBuffer<T>>,
    ) where
        T: Float;
}
