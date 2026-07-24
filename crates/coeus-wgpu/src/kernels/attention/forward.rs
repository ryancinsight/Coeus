// ── On-device WGPU scaled dot-product attention (forward) ──
//
// One invocation per `(batch, query)` row. Mirrors the verified CPU reference
// in `crates/coeus-ops/src/backend_ops/cpu_impl/attention.rs`: per-row scores ->
// numerically stable softmax -> attn·V. Tensors are contiguous `[batch, seq,
// dim]` with offset 0 (heads folded into batch). The attention matrix is
// materialized into `aw`, matching the CPU contract.
//
// Causal masking is handled by skipping `j > i` keys (their post-softmax weight
// is zero), which is exact and avoids relying on a WGSL infinity literal. The
// `key_padding_mask` case is handled by the CPU reference path (see
// `backend/ops/attention.rs`), not this kernel.

use crate::kernels::cache::PIPELINE_CACHE;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AttnParams {
    pub(crate) seq_q: u32,
    pub(crate) seq_k: u32,
    pub(crate) d_k: u32,
    pub(crate) d_v: u32,
    pub(crate) is_causal: u32,
    pub(crate) scale: f32,
    pub(crate) total: u32,
    pub(crate) has_mask: u32,
    pub(crate) mask_ndim: u32,
    pub(crate) num_heads: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
}

pub struct AttnForwardDispatch<'a> {
    pub query: &'a wgpu::Buffer,
    pub key: &'a wgpu::Buffer,
    pub value: &'a wgpu::Buffer,
    /// Contiguous key-padding mask (`None` for the unmasked case).
    pub mask: Option<&'a wgpu::Buffer>,
    pub output: &'a wgpu::Buffer,
    pub attn_weights: &'a wgpu::Buffer,
    pub batch: usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub is_causal: bool,
    pub scale: f32,
    pub mask_ndim: usize,
    pub num_heads: usize,
}

const SHADER: &str = r#"
struct AttnParams {
    seq_q: u32, seq_k: u32, d_k: u32, d_v: u32,
    is_causal: u32, scale: f32, total: u32, has_mask: u32,
    mask_ndim: u32, num_heads: u32, pad0: u32, pad1: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> aw: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<f32>;
@group(0) @binding(6) var<storage, read> params: AttnParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total) { return; }
    let seq_q = params.seq_q;
    let seq_k = params.seq_k;
    let d_k = params.d_k;
    let d_v = params.d_v;
    let b = idx / seq_q;
    let i = idx % seq_q;
    let q_base = (b * seq_q + i) * d_k;
    let aw_base = (b * seq_q + i) * seq_k;
    let out_base = (b * seq_q + i) * d_v;
    let k_base = b * seq_k * d_k;
    let v_base = b * seq_k * d_v;
    let causal = params.is_causal != 0u;
    let has_mask = params.has_mask != 0u;
    // Contiguous mask base: 2-D [batch_mask, seq_k] folds heads into batch.
    var mask_base = 0u;
    if (has_mask && params.mask_ndim == 2u) {
        mask_base = (b / params.num_heads) * seq_k;
    }

    // Pass 1: scores over valid keys; track row max.
    var mx = -3.4028235e38;
    for (var j: u32 = 0u; j < seq_k; j = j + 1u) {
        let masked_out = (causal && j > i) || (has_mask && mask[mask_base + j] == 0.0);
        if (masked_out) { aw[aw_base + j] = 0.0; continue; }
        var dot = 0.0;
        let kj = k_base + j * d_k;
        for (var d: u32 = 0u; d < d_k; d = d + 1u) {
            dot = fma(q[q_base + d], k[kj + d], dot);
        }
        let s = dot * params.scale;
        aw[aw_base + j] = s;
        if (s > mx) { mx = s; }
    }
    // Pass 2: exp(score - mx); accumulate denominator.
    var sum = 0.0;
    for (var j: u32 = 0u; j < seq_k; j = j + 1u) {
        let masked_out = (causal && j > i) || (has_mask && mask[mask_base + j] == 0.0);
        if (masked_out) { continue; }
        let e = exp(aw[aw_base + j] - mx);
        aw[aw_base + j] = e;
        sum = sum + e;
    }
    let inv = 1.0 / sum;
    // Pass 3: normalize valid weights.
    for (var j: u32 = 0u; j < seq_k; j = j + 1u) {
        let masked_out = (causal && j > i) || (has_mask && mask[mask_base + j] == 0.0);
        if (masked_out) { continue; }
        aw[aw_base + j] = aw[aw_base + j] * inv;
    }
    // Pass 4: out[i,l] = sum_j attn[i,j] * V[j,l] (masked weights are zero).
    for (var l: u32 = 0u; l < d_v; l = l + 1u) {
        var acc = 0.0;
        for (var j: u32 = 0u; j < seq_k; j = j + 1u) {
            acc = fma(aw[aw_base + j], v[v_base + j * d_v + l], acc);
        }
        out[out_base + l] = acc;
    }
}
"#;

pub fn dispatch_sdp_attention(request: AttnForwardDispatch<'_>) {
    let ctx = crate::backend::get_wgpu_context();
    let total = request.batch * request.seq_q;
    let params = AttnParams {
        seq_q: request.seq_q as u32,
        seq_k: request.seq_k as u32,
        d_k: request.d_k as u32,
        d_v: request.d_v as u32,
        is_causal: u32::from(request.is_causal),
        scale: request.scale,
        total: total as u32,
        has_mask: u32::from(request.mask.is_some()),
        mask_ndim: request.mask_ndim as u32,
        num_heads: request.num_heads.max(1) as u32,
        _pad0: 0,
        _pad1: 0,
    };

    let params_buf = crate::backend::PooledMetadataBuffer::new();
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    // Masked binding requires a valid storage buffer even when unused; the
    // shader never indexes it when `has_mask == 0`.
    let dummy_mask = crate::backend::PooledMetadataBuffer::new();
    let mask_buf = request.mask.unwrap_or(&dummy_mask);

    let pipeline = PIPELINE_CACHE.get_or_create("sdp_attn_fwd_f32", &ctx.device, SHADER, "main");
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sdp-attn-fwd-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: request.query.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: request.key.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: request.value.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: request.output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: request.attn_weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: mask_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sdp-attn-fwd-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sdp-attn-fwd-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = total.div_ceil(256);
        pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
}
