// ── CUDA SDP attention dispatch ──
//
// Routes scaled dot-product attention to the on-device NVRTC kernels
// (`kernels::attention`) for the f32 causal/unmasked case, and to the verified
// CPU reference (`fallback_sdp_attention`) otherwise. The masked case
// (`key_padding_mask.is_some()`) is an explicit capability boundary handled by
// the CPU path until a masked on-device kernel is added — not a silent fallback.

use super::cast::{cast_storage, cast_storage_mut};
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_sdp_attention<T: CudaScalar + coeus_core::Float>(
        &self,
        query: &CudaStorage<T>,
        query_layout: &Layout,
        key: &CudaStorage<T>,
        key_layout: &Layout,
        value: &CudaStorage<T>,
        value_layout: &Layout,
        key_padding_mask: Option<&CudaStorage<T>>,
        key_padding_mask_layout: Option<&Layout>,
        is_causal: bool,
        scale: T,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
        attn_weights: &mut CudaStorage<T>,
        attn_weights_layout: &Layout,
    ) {
        let on_device = key_padding_mask.is_none()
            && get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        if on_device {
            let q_shape = query_layout.shape();
            let batch = q_shape[0];
            let seq_q = q_shape[1];
            let d_k = q_shape[2];
            let seq_k = key_layout.shape()[1];
            let d_v = value_layout.shape()[2];
            // T is TypeId-confirmed f32 here; f32->f64->f32 round-trips exactly.
            let scale_f32 = coeus_core::Scalar::to_f64(scale) as f32;

            let q32 = cast_storage::<T, f32>(query);
            let k32 = cast_storage::<T, f32>(key);
            let v32 = cast_storage::<T, f32>(value);
            let mut out32 = cast_storage_mut::<T, f32>(output);
            let mut aw32 = cast_storage_mut::<T, f32>(attn_weights);

            if kernels::launch_sdp_attention(
                &q32, &k32, &v32, &mut out32, &mut aw32, batch, seq_q, seq_k, d_k, d_v, is_causal,
                scale_f32,
            ) {
                return;
            }
        }
        self.fallback_sdp_attention(
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
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_sdp_attention_backward<T: CudaScalar + coeus_core::Float>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        query: &CudaStorage<T>,
        query_layout: &Layout,
        key: &CudaStorage<T>,
        key_layout: &Layout,
        value: &CudaStorage<T>,
        value_layout: &Layout,
        attn_weights: &CudaStorage<T>,
        attn_weights_layout: &Layout,
        scale: T,
        mut grad_q: Option<&mut CudaStorage<T>>,
        mut grad_k: Option<&mut CudaStorage<T>>,
        mut grad_v: Option<&mut CudaStorage<T>>,
    ) {
        let on_device = get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        if on_device {
            let q_shape = query_layout.shape();
            let batch = q_shape[0];
            let seq_q = q_shape[1];
            let d_k = q_shape[2];
            let seq_k = key_layout.shape()[1];
            let d_v = value_layout.shape()[2];
            // T is TypeId-confirmed f32 here; f32->f64->f32 round-trips exactly.
            let scale_f32 = coeus_core::Scalar::to_f64(scale) as f32;

            let go32 = cast_storage::<T, f32>(grad_out);
            let q32 = cast_storage::<T, f32>(query);
            let k32 = cast_storage::<T, f32>(key);
            let v32 = cast_storage::<T, f32>(value);
            let aw32 = cast_storage::<T, f32>(attn_weights);

            let mut gq32 = grad_q.as_mut().map(|g| cast_storage_mut::<T, f32>(g));
            let mut gk32 = grad_k.as_mut().map(|g| cast_storage_mut::<T, f32>(g));
            let mut gv32 = grad_v.as_mut().map(|g| cast_storage_mut::<T, f32>(g));

            if kernels::launch_sdp_attention_backward(
                &go32,
                &q32,
                &k32,
                &v32,
                &aw32,
                gq32.as_mut(),
                gk32.as_mut(),
                gv32.as_mut(),
                batch,
                seq_q,
                seq_k,
                d_k,
                d_v,
                scale_f32,
            ) {
                return;
            }
        }
        self.fallback_sdp_attention_backward(
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
        );
    }
}
