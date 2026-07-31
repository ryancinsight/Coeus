use super::provider::{AttentionBackend, AttentionProvider};
use crate::HephaestusBackend;
use coeus_core::{Float, Layout, Scalar};
use hephaestus_core::AttentionScalar;

impl<P, T> coeus_ops::AttentionOps<T> for HephaestusBackend<P>
where
    P: AttentionProvider<T>,
    T: Scalar + Float + AttentionScalar + coeus_ops::AttentionScalar,
{
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
        T: Float,
    {
        self.dispatch_attention_forward(
            query,
            query_layout,
            key,
            key_layout,
            value,
            value_layout,
            key_padding_mask,
            key_padding_mask_layout,
            is_causal,
            scale,
            output,
            output_layout,
            attn_weights,
            attn_weights_layout,
        )
    }

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
        T: Float,
    {
        self.dispatch_attention_backward(
            grad_out,
            grad_out_layout,
            query,
            query_layout,
            key,
            key_layout,
            value,
            value_layout,
            attn_weights,
            attn_weights_layout,
            scale,
            grad_q,
            grad_k,
            grad_v,
        )
    }
}
