use crate::backend::{get_wgpu_context, WgpuBackend, WgpuBackendError};
use coeus_core::Layout;
use coeus_hephaestus::{AttentionBackend, AttentionProvider, HephaestusProvider};
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_wgpu::{WgpuAttentionOps, WgpuDevice};

// SAFETY: `WgpuStorage` retains the provider buffer behind `Arc`, and the
// process-global device owns queue synchronization for every submitted kernel.
unsafe impl HephaestusProvider for WgpuBackend {
    type Device = WgpuDevice;

    const NAME: &'static str = "wgpu";

    fn device() -> &'static Self::Device {
        &get_wgpu_context().hephaestus_device
    }
}

impl AttentionProvider<f32> for WgpuBackend {
    type Operations = WgpuAttentionOps;
}

impl AttentionBackend<f32> for WgpuBackend {
    type Provider = Self;

    fn attention_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<WgpuDevice as ComputeDevice>::Buffer<f32> {
        storage.buffer.as_ref()
    }

    fn attention_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }
}

impl coeus_ops::AttentionOps<f32> for WgpuBackend {
    fn sdp_attention(
        &self,
        query: &Self::DeviceBuffer<f32>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<f32>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<f32>,
        value_layout: &Layout,
        key_padding_mask: Option<&Self::DeviceBuffer<f32>>,
        key_padding_mask_layout: Option<&Layout>,
        is_causal: bool,
        scale: f32,
        output: &mut Self::DeviceBuffer<f32>,
        output_layout: &Layout,
        attn_weights: &mut Self::DeviceBuffer<f32>,
        attn_weights_layout: &Layout,
    ) -> Result<(), Self::Error> {
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
        grad_out: &Self::DeviceBuffer<f32>,
        grad_out_layout: &Layout,
        query: &Self::DeviceBuffer<f32>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<f32>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<f32>,
        value_layout: &Layout,
        attn_weights: &Self::DeviceBuffer<f32>,
        attn_weights_layout: &Layout,
        scale: f32,
        grad_q: Option<(&mut Self::DeviceBuffer<f32>, &Layout)>,
        grad_k: Option<(&mut Self::DeviceBuffer<f32>, &Layout)>,
        grad_v: Option<(&mut Self::DeviceBuffer<f32>, &Layout)>,
    ) -> Result<(), Self::Error> {
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
