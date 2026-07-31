use crate::backend::{get_cuda_device, CudaBackend, CudaScalar};
use crate::CudaBackendError;
use coeus_core::{Float, Layout};
use coeus_hephaestus::{AttentionBackend, AttentionProvider, HephaestusProvider};
use hephaestus_core::{AttentionOps, AttentionScalar, ComputeDevice, HephaestusError};
use hephaestus_cuda::{CudaAttentionOps, CudaDevice};

// SAFETY: `CudaStorage` retains provider buffers behind `Arc`, and the
// process-global CUDA device owns stream synchronization for every dispatch.
unsafe impl HephaestusProvider for CudaBackend {
    type Device = CudaDevice;

    const NAME: &'static str = "cuda";

    fn device() -> &'static Self::Device {
        get_cuda_device()
    }
}

impl<T> AttentionProvider<T> for CudaBackend
where
    T: CudaScalar + Float + AttentionScalar + coeus_ops::AttentionScalar,
    CudaAttentionOps: AttentionOps<CudaDevice, T>,
{
    type Operations = CudaAttentionOps;
}

impl<T> AttentionBackend<T> for CudaBackend
where
    T: CudaScalar + Float + AttentionScalar + coeus_ops::AttentionScalar,
    CudaAttentionOps: AttentionOps<CudaDevice, T>,
{
    type Provider = Self;

    fn attention_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<CudaDevice as ComputeDevice>::Buffer<T> {
        storage.buffer.as_ref()
    }

    fn attention_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        CudaBackendError::dispatch(operation, source)
    }
}

impl<T> coeus_ops::AttentionOps<T> for CudaBackend
where
    T: CudaScalar + Float + AttentionScalar + coeus_ops::AttentionScalar,
    CudaAttentionOps: AttentionOps<CudaDevice, T>,
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
