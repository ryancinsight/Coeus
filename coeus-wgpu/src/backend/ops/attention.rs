use crate::backend::WgpuBackend;
use coeus_core::{
    ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float, Layout, SequentialBackend,
    Storage,
};
use coeus_ops::BackendOps;

pub fn sdp_attention<T: Float + leto_ops::Scalar>(
    backend: &WgpuBackend,
    query: &<WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
    query_layout: &Layout,
    key: &<WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
    key_layout: &Layout,
    value: &<WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
    value_layout: &Layout,
    key_padding_mask: Option<&<WgpuBackend as ComputeBackend>::DeviceBuffer<T>>,
    key_padding_mask_layout: Option<&Layout>,
    is_causal: bool,
    scale: T,
    output: &mut <WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
    output_layout: &Layout,
    attn_weights: &mut <WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
    attn_weights_layout: &Layout,
) {
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

#[allow(clippy::too_many_arguments)]
pub fn sdp_attention_backward<T: Float + leto_ops::Scalar>(
    backend: &WgpuBackend,
    grad_out: &<WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
    grad_out_layout: &Layout,
    query: &<WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
    query_layout: &Layout,
    key: &<WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
    key_layout: &Layout,
    value: &<WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
    value_layout: &Layout,
    attn_weights: &<WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
    attn_weights_layout: &Layout,
    scale: T,
    grad_q: Option<&mut <WgpuBackend as ComputeBackend>::DeviceBuffer<T>>,
    grad_k: Option<&mut <WgpuBackend as ComputeBackend>::DeviceBuffer<T>>,
    grad_v: Option<&mut <WgpuBackend as ComputeBackend>::DeviceBuffer<T>>,
) {
    let seq = SequentialBackend::new();
    let go_len = grad_out.len();
    let q_len = query.len();
    let k_len = key.len();
    let v_len = value.len();
    let aw_len = attn_weights.len();
    let mut go_cpu = seq.allocate::<T>(go_len);
    let mut q_cpu = seq.allocate::<T>(q_len);
    let mut k_cpu = seq.allocate::<T>(k_len);
    let mut v_cpu = seq.allocate::<T>(v_len);
    let mut aw_cpu = seq.allocate::<T>(aw_len);
    backend.copy_to_host(grad_out, go_cpu.as_mut_slice());
    backend.copy_to_host(query, q_cpu.as_mut_slice());
    backend.copy_to_host(key, k_cpu.as_mut_slice());
    backend.copy_to_host(value, v_cpu.as_mut_slice());
    backend.copy_to_host(attn_weights, aw_cpu.as_mut_slice());

    let mut gq_cpu = grad_q.as_ref().map(|g| {
        let mut s = seq.allocate::<T>(g.len());
        backend.copy_to_host(*g, s.as_mut_slice());
        s
    });
    let mut gk_cpu = grad_k.as_ref().map(|g| {
        let mut s = seq.allocate::<T>(g.len());
        backend.copy_to_host(*g, s.as_mut_slice());
        s
    });
    let mut gv_cpu = grad_v.as_ref().map(|g| {
        let mut s = seq.allocate::<T>(g.len());
        backend.copy_to_host(*g, s.as_mut_slice());
        s
    });

    seq.sdp_attention_backward(
        &go_cpu,
        grad_out_layout,
        &q_cpu,
        query_layout,
        &k_cpu,
        key_layout,
        &v_cpu,
        value_layout,
        &aw_cpu,
        attn_weights_layout,
        scale,
        gq_cpu.as_mut(),
        gk_cpu.as_mut(),
        gv_cpu.as_mut(),
    );

    if let (Some(gq_gpu), Some(ref gq_c)) = (grad_q, &gq_cpu) {
        backend.copy_to_device(gq_c.as_slice(), gq_gpu);
    }
    if let (Some(gk_gpu), Some(ref gk_c)) = (grad_k, &gk_cpu) {
        backend.copy_to_device(gk_c.as_slice(), gk_gpu);
    }
    if let (Some(gv_gpu), Some(ref gv_c)) = (grad_v, &gv_cpu) {
        backend.copy_to_device(gv_c.as_slice(), gv_gpu);
    }
}
