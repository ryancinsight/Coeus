// ── On-device WGPU scaled dot-product attention (backward) ──
//
// Mirrors the CPU reference: given stored post-softmax weights `A` and `grad_out`
//   dV = A^T @ dO,   dA = dO @ V^T,   dS = A ⊙ (dA - rowsum(A ⊙ dA))
//   dQ = dS @ K * scale,   dK = dS^T @ Q * scale
//
// Two stream-ordered passes share a transient `d_scores` device buffer:
//   pass 1 (one invocation per query row): fill `d_scores`, accumulate dQ;
//   pass 2 (one invocation per key row):   accumulate dK and dV.
// The backward is mask-agnostic — masked positions carry `A = 0`, so they
// contribute nothing — hence no causal/mask parameter is needed here.

use crate::kernels::cache::PIPELINE_CACHE;
use crate::storage::WgpuStorage;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BwdParams {
    seq_q: u32,
    seq_k: u32,
    d_k: u32,
    d_v: u32,
    flag_a: u32,
    flag_b: u32,
    scale: f32,
    total: u32,
}

pub struct AttnBackwardDispatch<'a> {
    pub grad_out: &'a wgpu::Buffer,
    pub query: &'a wgpu::Buffer,
    pub key: &'a wgpu::Buffer,
    pub value: &'a wgpu::Buffer,
    pub attn_weights: &'a wgpu::Buffer,
    pub grad_q: Option<&'a wgpu::Buffer>,
    pub grad_k: Option<&'a wgpu::Buffer>,
    pub grad_v: Option<&'a wgpu::Buffer>,
    pub batch: usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub scale: f32,
}

const DQ_SHADER: &str = r#"
struct P { seq_q: u32, seq_k: u32, d_k: u32, d_v: u32, has_gq: u32, fb: u32, scale: f32, total: u32, }

@group(0) @binding(0) var<storage, read> go: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> aw: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_scores: array<f32>;
@group(0) @binding(5) var<storage, read_write> gq: array<f32>;
@group(0) @binding(6) var<storage, read> params: P;

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
    let go_base = (b * seq_q + i) * d_v;
    let aw_base = (b * seq_q + i) * seq_k;
    let ds_base = (b * seq_q + i) * seq_k;
    let v_base = b * seq_k * d_v;
    let k_base = b * seq_k * d_k;

    // d_attn_row[j] = dot(dO[i,:], V[j,:]) -> stash in d_scores row.
    for (var j: u32 = 0u; j < seq_k; j = j + 1u) {
        var dot = 0.0;
        let vj = v_base + j * d_v;
        for (var l: u32 = 0u; l < d_v; l = l + 1u) {
            dot = fma(go[go_base + l], v[vj + l], dot);
        }
        d_scores[ds_base + j] = dot;
    }
    // rs = dot(A[i,:], d_attn_row).
    var rs = 0.0;
    for (var j: u32 = 0u; j < seq_k; j = j + 1u) {
        rs = fma(aw[aw_base + j], d_scores[ds_base + j], rs);
    }
    // d_scores[i,j] = A[i,j] * (d_attn_row[j] - rs).
    for (var j: u32 = 0u; j < seq_k; j = j + 1u) {
        d_scores[ds_base + j] = aw[aw_base + j] * (d_scores[ds_base + j] - rs);
    }
    // dQ[i,d] += scale * sum_j d_scores[i,j] * K[j,d].
    if (params.has_gq != 0u) {
        let gq_base = (b * seq_q + i) * d_k;
        for (var d: u32 = 0u; d < d_k; d = d + 1u) {
            var acc = 0.0;
            for (var j: u32 = 0u; j < seq_k; j = j + 1u) {
                acc = fma(d_scores[ds_base + j], k[k_base + j * d_k + d], acc);
            }
            gq[gq_base + d] = gq[gq_base + d] + acc * params.scale;
        }
    }
}
"#;

const DKV_SHADER: &str = r#"
struct P { seq_q: u32, seq_k: u32, d_k: u32, d_v: u32, has_gk: u32, has_gv: u32, scale: f32, total: u32, }

@group(0) @binding(0) var<storage, read> go: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> aw: array<f32>;
@group(0) @binding(3) var<storage, read> d_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> gk: array<f32>;
@group(0) @binding(5) var<storage, read_write> gv: array<f32>;
@group(0) @binding(6) var<storage, read> params: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total) { return; }
    let seq_q = params.seq_q;
    let seq_k = params.seq_k;
    let d_k = params.d_k;
    let d_v = params.d_v;
    let b = idx / seq_k;
    let j = idx % seq_k;

    // dK[j,d] += scale * sum_i d_scores[i,j] * Q[i,d].
    if (params.has_gk != 0u) {
        let gk_base = (b * seq_k + j) * d_k;
        for (var d: u32 = 0u; d < d_k; d = d + 1u) {
            var acc = 0.0;
            for (var i: u32 = 0u; i < seq_q; i = i + 1u) {
                let ds = d_scores[(b * seq_q + i) * seq_k + j];
                let qv = q[(b * seq_q + i) * d_k + d];
                acc = fma(ds, qv, acc);
            }
            gk[gk_base + d] = gk[gk_base + d] + acc * params.scale;
        }
    }
    // dV[j,l] += sum_i A[i,j] * dO[i,l].
    if (params.has_gv != 0u) {
        let gv_base = (b * seq_k + j) * d_v;
        for (var l: u32 = 0u; l < d_v; l = l + 1u) {
            var acc = 0.0;
            for (var i: u32 = 0u; i < seq_q; i = i + 1u) {
                let awv = aw[(b * seq_q + i) * seq_k + j];
                let gov = go[(b * seq_q + i) * d_v + l];
                acc = fma(awv, gov, acc);
            }
            gv[gv_base + l] = gv[gv_base + l] + acc;
        }
    }
}
"#;

pub fn dispatch_sdp_attention_backward(
    request: AttnBackwardDispatch<'_>,
) -> Result<(), crate::backend::WgpuBackendError> {
    let ctx = crate::backend::get_wgpu_context();
    let scale = request.scale;
    let (batch, seq_q, seq_k, d_k, d_v) = (
        request.batch,
        request.seq_q,
        request.seq_k,
        request.d_k,
        request.d_v,
    );

    // Transient scratch for d_scores (also doubles as the dummy grad binding).
    let d_scores = WgpuStorage::<f32>::try_new(batch * seq_q * seq_k)?;
    let dummy = crate::backend::PooledMetadataBuffer::new();

    // ── Pass 1: fill d_scores, accumulate dQ (one invocation per query row). ──
    {
        let total_q = batch * seq_q;
        let params = BwdParams {
            seq_q: seq_q as u32,
            seq_k: seq_k as u32,
            d_k: d_k as u32,
            d_v: d_v as u32,
            flag_a: u32::from(request.grad_q.is_some()),
            flag_b: 0,
            scale,
            total: total_q as u32,
        };
        let params_buf = crate::backend::PooledMetadataBuffer::new();
        ctx.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
        let gq = request.grad_q.unwrap_or(&dummy);

        let pipeline =
            PIPELINE_CACHE.get_or_create("sdp_attn_bwd_dq_f32", &ctx.device, DQ_SHADER, "main");
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdp-attn-bwd-dq-bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: request.grad_out.as_entire_binding(),
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
                    resource: request.attn_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: d_scores.buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gq.as_entire_binding(),
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
                label: Some("sdp-attn-bwd-dq-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdp-attn-bwd-dq-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_q.div_ceil(256) as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    // ── Pass 2: accumulate dK and dV (one invocation per key row). ──
    {
        let total_k = batch * seq_k;
        let params = BwdParams {
            seq_q: seq_q as u32,
            seq_k: seq_k as u32,
            d_k: d_k as u32,
            d_v: d_v as u32,
            flag_a: u32::from(request.grad_k.is_some()),
            flag_b: u32::from(request.grad_v.is_some()),
            scale,
            total: total_k as u32,
        };
        let params_buf = crate::backend::PooledMetadataBuffer::new();
        ctx.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
        let gk = request.grad_k.unwrap_or(&dummy);
        let gv = request.grad_v.unwrap_or(&dummy);

        let pipeline =
            PIPELINE_CACHE.get_or_create("sdp_attn_bwd_dkv_f32", &ctx.device, DKV_SHADER, "main");
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdp-attn-bwd-dkv-bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: request.grad_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: request.query.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: request.attn_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: d_scores.buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gk.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gv.as_entire_binding(),
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
                label: Some("sdp-attn-bwd-dkv-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdp-attn-bwd-dkv-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_k.div_ceil(256) as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }
    Ok(())
}
