// ── On-device transposed convolution forward (WGSL) ──
//
// Mirrors the verified CPU reference (`BackendOps::conv_transpose1d`/`2d`). The
// CPU path scatters each input element to outputs; these shaders use the
// equivalent gather — one invocation per output element accumulates its
// contributions — which is conflict-free on the GPU (no atomics).
//
// Contiguous, offset 0:
//   1d: input [n, c_in, l],  weight [c_in, c_out, k],  output [n, c_out, l_out]
//   2d: input [n, c_in, h, w], weight [c_in, c_out, kh, kw], output [n, c_out, h_out, w_out]
//
// For output position p the contributing input index satisfies
//   t*stride = p + padding - ki*dilation   (must be exact and in-range),
// the gather inverse of the scatter `p = t*stride + ki*dilation - padding`.

use crate::kernels::cache::PIPELINE_CACHE;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CtParams1d {
    n: u32,
    c_in: u32,
    l: u32,
    c_out: u32,
    k: u32,
    l_out: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    has_bias: u32,
    total: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CtParams2d {
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kh: u32,
    kw: u32,
    h_out: u32,
    w_out: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    has_bias: u32,
    total: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct ConvTranspose1dDispatch<'a> {
    pub input: &'a wgpu::Buffer,
    pub weight: &'a wgpu::Buffer,
    pub bias: Option<&'a wgpu::Buffer>,
    pub output: &'a wgpu::Buffer,
    pub n: usize,
    pub c_in: usize,
    pub l: usize,
    pub c_out: usize,
    pub k: usize,
    pub l_out: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

pub struct ConvTranspose2dDispatch<'a> {
    pub input: &'a wgpu::Buffer,
    pub weight: &'a wgpu::Buffer,
    pub bias: Option<&'a wgpu::Buffer>,
    pub output: &'a wgpu::Buffer,
    pub n: usize,
    pub c_in: usize,
    pub h: usize,
    pub w: usize,
    pub c_out: usize,
    pub kh: usize,
    pub kw: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

const SHADER_1D: &str = r#"
struct P {
    n: u32, c_in: u32, l: u32, c_out: u32, k: u32, l_out: u32,
    stride: u32, padding: u32, dilation: u32, has_bias: u32, total: u32, pad: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.total) { return; }
    let t_out = idx % p.l_out;
    let tmp = idx / p.l_out;
    let oc = tmp % p.c_out;
    let ni = tmp / p.c_out;

    var acc = 0.0;
    for (var ic: u32 = 0u; ic < p.c_in; ic = ic + 1u) {
        let in_base = (ni * p.c_in + ic) * p.l;
        let w_base = (ic * p.c_out + oc) * p.k;
        for (var ki: u32 = 0u; ki < p.k; ki = ki + 1u) {
            let num = i32(t_out + p.padding) - i32(ki * p.dilation);
            if (num < 0) { continue; }
            if (u32(num) % p.stride != 0u) { continue; }
            let ti = u32(num) / p.stride;
            if (ti >= p.l) { continue; }
            acc = fma(input[in_base + ti], weight[w_base + ki], acc);
        }
    }
    if (p.has_bias != 0u) { acc = acc + bias[oc]; }
    out[(ni * p.c_out + oc) * p.l_out + t_out] = acc;
}
"#;

const SHADER_2D: &str = r#"
struct P {
    n: u32, c_in: u32, h: u32, w: u32, c_out: u32, kh: u32, kw: u32,
    h_out: u32, w_out: u32, stride: u32, padding: u32, dilation: u32,
    has_bias: u32, total: u32, pad0: u32, pad1: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.total) { return; }
    let wo = idx % p.w_out;
    let t1 = idx / p.w_out;
    let ho = t1 % p.h_out;
    let t2 = t1 / p.h_out;
    let oc = t2 % p.c_out;
    let ni = t2 / p.c_out;

    var acc = 0.0;
    for (var ic: u32 = 0u; ic < p.c_in; ic = ic + 1u) {
        let in_plane = (ni * p.c_in + ic) * p.h * p.w;
        let w_plane = (ic * p.c_out + oc) * p.kh * p.kw;
        for (var ki: u32 = 0u; ki < p.kh; ki = ki + 1u) {
            let numh = i32(ho + p.padding) - i32(ki * p.dilation);
            if (numh < 0) { continue; }
            if (u32(numh) % p.stride != 0u) { continue; }
            let hi = u32(numh) / p.stride;
            if (hi >= p.h) { continue; }
            for (var kj: u32 = 0u; kj < p.kw; kj = kj + 1u) {
                let numw = i32(wo + p.padding) - i32(kj * p.dilation);
                if (numw < 0) { continue; }
                if (u32(numw) % p.stride != 0u) { continue; }
                let wi = u32(numw) / p.stride;
                if (wi >= p.w) { continue; }
                acc = fma(input[in_plane + hi * p.w + wi], weight[w_plane + ki * p.kw + kj], acc);
            }
        }
    }
    if (p.has_bias != 0u) { acc = acc + bias[oc]; }
    out[(ni * p.c_out + oc) * p.h_out * p.w_out + ho * p.w_out + wo] = acc;
}
"#;

#[allow(clippy::too_many_arguments)]
fn run(
    label: &str,
    shader: &str,
    key: &str,
    input: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    output: &wgpu::Buffer,
    params_bytes: &[u8],
    total: usize,
) {
    let ctx = crate::backend::get_wgpu_context();
    let params_buf = crate::backend::PooledMetadataBuffer::new();
    ctx.queue.write_buffer(&params_buf, 0, params_bytes);
    let dummy = crate::backend::PooledMetadataBuffer::new();
    let bias_buf = bias.unwrap_or(&dummy);

    let pipeline = PIPELINE_CACHE.get_or_create(key, &ctx.device, shader, "main");
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bias_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(total.div_ceil(256) as u32, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
}

pub fn dispatch_conv_transpose1d(req: ConvTranspose1dDispatch<'_>) {
    let total = req.n * req.c_out * req.l_out;
    let params = CtParams1d {
        n: req.n as u32,
        c_in: req.c_in as u32,
        l: req.l as u32,
        c_out: req.c_out as u32,
        k: req.k as u32,
        l_out: req.l_out as u32,
        stride: req.stride as u32,
        padding: req.padding as u32,
        dilation: req.dilation as u32,
        has_bias: u32::from(req.bias.is_some()),
        total: total as u32,
        _pad: 0,
    };
    run(
        "conv_transpose1d",
        SHADER_1D,
        "conv_transpose1d_f32",
        req.input,
        req.weight,
        req.bias,
        req.output,
        bytemuck::bytes_of(&params),
        total,
    );
}

pub fn dispatch_conv_transpose2d(req: ConvTranspose2dDispatch<'_>) {
    let total = req.n * req.c_out * req.h_out * req.w_out;
    let params = CtParams2d {
        n: req.n as u32,
        c_in: req.c_in as u32,
        h: req.h as u32,
        w: req.w as u32,
        c_out: req.c_out as u32,
        kh: req.kh as u32,
        kw: req.kw as u32,
        h_out: req.h_out as u32,
        w_out: req.w_out as u32,
        stride: req.stride as u32,
        padding: req.padding as u32,
        dilation: req.dilation as u32,
        has_bias: u32::from(req.bias.is_some()),
        total: total as u32,
        _pad0: 0,
        _pad1: 0,
    };
    run(
        "conv_transpose2d",
        SHADER_2D,
        "conv_transpose2d_f32",
        req.input,
        req.weight,
        req.bias,
        req.output,
        bytemuck::bytes_of(&params),
        total,
    );
}
