// ── CUDA SDP attention dispatch ──
//
// Routes scaled dot-product attention to the on-device NVRTC kernels
// (`kernels::attention`) for contiguous f32 tensors, including rank-1 and
// rank-2 key-padding masks. Unsupported layouts and mask shapes return a
// typed capability error instead of changing execution backends.

use super::cast::{cast_storage, cast_storage_mut};
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::error::CudaBackendError;
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
    ) -> Result<(), CudaBackendError> {
        let layouts_on_device = [
            query_layout,
            key_layout,
            value_layout,
            output_layout,
            attn_weights_layout,
        ]
        .into_iter()
        .all(|layout| layout.ndim() == 3 && layout.is_contiguous() && layout.offset() == 0);
        let on_device = layouts_on_device
            && get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        if on_device {
            let q_shape = query_layout.shape();
            let batch = q_shape[0];
            let seq_q = q_shape[1];
            let d_k = q_shape[2];
            let seq_k = key_layout.shape()[1];
            let d_v = value_layout.shape()[2];
            let shapes_match = key_layout.shape()[0] == batch
                && key_layout.shape()[2] == d_k
                && value_layout.shape()[0] == batch
                && value_layout.shape()[1] == seq_k
                && output_layout.shape() == [batch, seq_q, d_v]
                && attn_weights_layout.shape() == [batch, seq_q, seq_k];
            if !shapes_match {
                return Err(CudaBackendError::dispatch_unavailable(
                    "sdp_attention",
                    "query, key, value, and output shapes are incompatible with the CUDA kernel",
                ));
            }
            // T is TypeId-confirmed f32 here; f32->f64->f32 round-trips exactly.
            let scale_f32 = coeus_core::Scalar::to_f64(scale) as f32;

            let mask_info = match (key_padding_mask, key_padding_mask_layout) {
                (None, None) => Some((0, 1)),
                (Some(_), Some(mask_layout))
                    if mask_layout.is_contiguous() && mask_layout.offset() == 0 =>
                {
                    match mask_layout.shape() {
                        [mask_seq] if *mask_seq == seq_k => Some((1, 1)),
                        [mask_batch, mask_seq]
                            if *mask_batch != 0
                                && *mask_seq == seq_k
                                && batch.is_multiple_of(*mask_batch) =>
                        {
                            Some((2, batch / *mask_batch))
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            let Some((mask_ndim, num_heads)) = mask_info else {
                return Err(CudaBackendError::dispatch_unavailable(
                    "sdp_attention",
                    "key-padding mask must be contiguous rank-1 or rank-2 data matching key length",
                ));
            };

            let q32 = cast_storage::<T, f32>(query);
            let k32 = cast_storage::<T, f32>(key);
            let v32 = cast_storage::<T, f32>(value);
            let mask32 = key_padding_mask.map(|m| cast_storage::<T, f32>(m));
            let mut out32 = cast_storage_mut::<T, f32>(output);
            let mut aw32 = cast_storage_mut::<T, f32>(attn_weights);

            if kernels::launch_sdp_attention(
                &q32,
                &k32,
                &v32,
                mask32.as_ref(),
                &mut out32,
                &mut aw32,
                batch,
                seq_q,
                seq_k,
                d_k,
                d_v,
                is_causal,
                scale_f32,
                mask_ndim,
                num_heads,
            ) {
                return Ok(());
            }
        }
        Err(CudaBackendError::dispatch_unavailable(
            "sdp_attention",
            "native CUDA dispatch requires an initialized context and contiguous f32 layouts",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_sdp_attention_backward<T: CudaScalar + coeus_core::Float>(
        &self,
        grad_out: &CudaStorage<T>,
        _grad_out_layout: &Layout,
        query: &CudaStorage<T>,
        query_layout: &Layout,
        key: &CudaStorage<T>,
        key_layout: &Layout,
        value: &CudaStorage<T>,
        value_layout: &Layout,
        attn_weights: &CudaStorage<T>,
        _attn_weights_layout: &Layout,
        scale: T,
        mut grad_q: Option<&mut CudaStorage<T>>,
        mut grad_k: Option<&mut CudaStorage<T>>,
        mut grad_v: Option<&mut CudaStorage<T>>,
    ) -> Result<(), CudaBackendError> {
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
                return Ok(());
            }
        }
        Err(CudaBackendError::dispatch_unavailable(
            "sdp_attention_backward",
            "native CUDA dispatch requires an initialized context and supported f32 layouts",
        ))
    }
}
