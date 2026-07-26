use crate::backend::WgpuBackend;
use coeus_core::{
    ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float, Layout,
    SequentialBackend, Storage,
};
use coeus_ops::AttentionOps;

type WgpuBuffer<T> = <WgpuBackend as ComputeBackend>::DeviceBuffer<T>;

pub struct AttentionForward<'a, T: Float> {
    pub backend: &'a WgpuBackend,
    pub query: &'a WgpuBuffer<T>,
    pub query_layout: &'a Layout,
    pub key: &'a WgpuBuffer<T>,
    pub key_layout: &'a Layout,
    pub value: &'a WgpuBuffer<T>,
    pub value_layout: &'a Layout,
    pub key_padding_mask: Option<&'a WgpuBuffer<T>>,
    pub key_padding_mask_layout: Option<&'a Layout>,
    pub is_causal: bool,
    pub scale: T,
    pub output: &'a mut WgpuBuffer<T>,
    pub output_layout: &'a Layout,
    pub attn_weights: &'a mut WgpuBuffer<T>,
    pub attn_weights_layout: &'a Layout,
}

pub struct AttentionBackward<'a, T: Float> {
    pub grad_out: &'a WgpuBuffer<T>,
    pub query: &'a WgpuBuffer<T>,
    pub query_layout: &'a Layout,
    pub key: &'a WgpuBuffer<T>,
    pub key_layout: &'a Layout,
    pub value: &'a WgpuBuffer<T>,
    pub value_layout: &'a Layout,
    pub attn_weights: &'a WgpuBuffer<T>,
    pub scale: T,
    pub grad_q: Option<&'a mut WgpuBuffer<T>>,
    pub grad_k: Option<&'a mut WgpuBuffer<T>>,
    pub grad_v: Option<&'a mut WgpuBuffer<T>>,
}

pub fn sdp_attention<T: Float + leto_ops::Scalar>(request: AttentionForward<'_, T>) {
    // The unmasked (causal or full) case runs on-device. The key_padding_mask
    // case is an explicit CPU-reference capability boundary, not a silent
    // fallback (see module note in `kernels/attention/forward.rs`).
    // A contiguous key-padding mask is handled on-device; a strided mask (rare)
    // routes to the CPU reference, an explicit capability boundary.
    let mask_on_device = request
        .key_padding_mask_layout
        .is_none_or(Layout::is_contiguous);
    if mask_on_device {
        let q_shape = request.query_layout.shape();
        let batch = q_shape[0];
        let seq_q = q_shape[1];
        let d_k = q_shape[2];
        let seq_k = request.key_layout.shape()[1];
        let d_v = request.value_layout.shape()[2];
        // T is Float + WgpuScalar, i.e. f32; f32->f64->f32 round-trips exactly.
        let scale_f32 = coeus_core::Scalar::to_f64(request.scale) as f32;
        let (mask_ndim, num_heads) = match request.key_padding_mask_layout {
            Some(ml) => {
                let nd = ml.ndim();
                let batch_mask = if nd == 2 { ml.shape()[0] } else { 1 };
                (nd, batch / batch_mask.max(1))
            }
            None => (0, 1),
        };
        crate::kernels::dispatch_sdp_attention(crate::kernels::AttnForwardDispatch {
            query: request.query.buffer.raw(),
            key: request.key.buffer.raw(),
            value: request.value.buffer.raw(),
            mask: request.key_padding_mask.map(|m| m.buffer.raw()),
            output: request.output.buffer.raw(),
            attn_weights: request.attn_weights.buffer.raw(),
            batch,
            seq_q,
            seq_k,
            d_k,
            d_v,
            is_causal: request.is_causal,
            scale: scale_f32,
            mask_ndim,
            num_heads,
        });
        return;
    }
    sdp_attention_cpu(request);
}

fn sdp_attention_cpu<T: Float + leto_ops::Scalar>(request: AttentionForward<'_, T>) {
    let AttentionForward {
        backend,
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
    } = request;
    let seq = SequentialBackend::new();
    let q_len = query.len();
    let k_len = key.len();
    let v_len = value.len();
    let out_len = output.len();
    let aw_len = attn_weights.len();
    let mut q_cpu = seq.allocate::<T>(q_len);
    let mut k_cpu = seq.allocate::<T>(k_len);
    let mut v_cpu = seq.allocate::<T>(v_len);
    let mut out_cpu = seq.allocate::<T>(out_len);
    let mut aw_cpu = seq.allocate::<T>(aw_len);
    backend.copy_to_host(query, q_cpu.as_mut_slice());
    backend.copy_to_host(key, k_cpu.as_mut_slice());
    backend.copy_to_host(value, v_cpu.as_mut_slice());

    let mask_cpu = key_padding_mask.map(|mask| {
        let mut buf = seq.allocate::<T>(mask.len());
        backend.copy_to_host(mask, buf.as_mut_slice());
        buf
    });

    seq.sdp_attention(
        &q_cpu,
        query_layout,
        &k_cpu,
        key_layout,
        &v_cpu,
        value_layout,
        mask_cpu.as_ref(),
        key_padding_mask_layout,
        is_causal,
        scale,
        &mut out_cpu,
        output_layout,
        &mut aw_cpu,
        attn_weights_layout,
    );

    backend.copy_to_device(out_cpu.as_slice(), output);
    backend.copy_to_device(aw_cpu.as_slice(), attn_weights);
}

pub fn sdp_attention_backward<T: Float + leto_ops::Scalar>(request: AttentionBackward<'_, T>) {
    // The backward is mask-agnostic: masked positions carry A = 0 in the stored
    // `attn_weights`, so they contribute nothing. It therefore always runs
    // on-device regardless of how the forward was produced.
    let AttentionBackward {
        grad_out,
        query,
        query_layout,
        key,
        key_layout,
        value,
        value_layout,
        attn_weights,
        scale,
        grad_q,
        grad_k,
        grad_v,
    } = request;

    let q_shape = query_layout.shape();
    let batch = q_shape[0];
    let seq_q = q_shape[1];
    let d_k = q_shape[2];
    let seq_k = key_layout.shape()[1];
    let d_v = value_layout.shape()[2];
    // T is Float + WgpuScalar, i.e. f32; f32->f64->f32 round-trips exactly.
    let scale_f32 = coeus_core::Scalar::to_f64(scale) as f32;

    crate::kernels::dispatch_sdp_attention_backward(crate::kernels::AttnBackwardDispatch {
        grad_out: grad_out.buffer.raw(),
        query: query.buffer.raw(),
        key: key.buffer.raw(),
        value: value.buffer.raw(),
        attn_weights: attn_weights.buffer.raw(),
        grad_q: grad_q.map(|g| g.buffer.raw()),
        grad_k: grad_k.map(|g| g.buffer.raw()),
        grad_v: grad_v.map(|g| g.buffer.raw()),
        batch,
        seq_q,
        seq_k,
        d_k,
        d_v,
        scale: scale_f32,
    });
}
