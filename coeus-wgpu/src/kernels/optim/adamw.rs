#![allow(clippy::too_many_arguments)]

use crate::backend::WgpuScalar;
use coeus_core::Layout;

use crate::kernels::cache::PIPELINE_CACHE;
use crate::kernels::layout::GpuLayoutInfo;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AdamWParams<T> {
    pub lr: T,
    pub beta1: T,
    pub beta2: T,
    pub eps: T,
    pub weight_decay: T,
    pub bias_correction1: T,
    pub bias_correction2: T,
}
unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for AdamWParams<T> {}
unsafe impl<T: bytemuck::Pod> bytemuck::Pod for AdamWParams<T> {}

pub fn dispatch_adamw_step<T: WgpuScalar + coeus_core::Float>(
    param: &wgpu::Buffer,
    param_layout: &Layout,
    grad: &wgpu::Buffer,
    grad_layout: &Layout,
    m: &wgpu::Buffer,
    m_layout: &Layout,
    v: &wgpu::Buffer,
    v_layout: &Layout,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    weight_decay: T,
    t: usize,
    len: usize,
) {
    let ctx = crate::backend::get_wgpu_context();

    let p_layout_gpu = GpuLayoutInfo::from_layout(param_layout);
    let g_layout_gpu = GpuLayoutInfo::from_layout(grad_layout);
    let m_layout_gpu = GpuLayoutInfo::from_layout(m_layout);
    let v_layout_gpu = GpuLayoutInfo::from_layout(v_layout);

    let t_float = T::from_f64(t as f64);
    let bias_correction1 = T::one() - beta1.powf(t_float);
    let bias_correction2 = T::one() - beta2.powf(t_float);

    let params_data = AdamWParams {
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
    };

    let layouts_data = [p_layout_gpu, g_layout_gpu, m_layout_gpu, v_layout_gpu];

    let layouts_buf = crate::backend::PooledMetadataBuffer::new();
    let params_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue
        .write_buffer(&layouts_buf, 0, bytemuck::cast_slice(&layouts_data));
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params_data));

    let wgsl_type = T::WGSL_TYPE;

    let shader_src = format!(
        r#"
        struct LayoutInfo {{
            offset: u32,
            ndim: u32,
            shape: array<u32, 8>,
            strides: array<u32, 8>,
        }}

        struct AdamWParams {{
            lr: {wgsl_type},
            beta1: {wgsl_type},
            beta2: {wgsl_type},
            eps: {wgsl_type},
            weight_decay: {wgsl_type},
            bias_correction1: {wgsl_type},
            bias_correction2: {wgsl_type},
        }}

        @group(0) @binding(0) var<storage, read_write> param: array<{wgsl_type}>;
        @group(0) @binding(1) var<storage, read> grad: array<{wgsl_type}>;
        @group(0) @binding(2) var<storage, read_write> m: array<{wgsl_type}>;
        @group(0) @binding(3) var<storage, read_write> v: array<{wgsl_type}>;
        @group(0) @binding(4) var<storage, read> layouts: array<LayoutInfo, 4>;
        @group(0) @binding(5) var<storage, read> params: AdamWParams;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let idx = global_id.x;
            if (idx >= arrayLength(&param)) {{
                return;
            }}

            var temp = idx;
            var off_p = layouts[0u].offset;
            var off_g = layouts[1u].offset;
            var off_m = layouts[2u].offset;
            var off_v = layouts[3u].offset;

            var p_contig_strides: array<u32, 8>;
            var accum: u32 = 1u;
            for (var d: i32 = i32(layouts[0u].ndim) - 1; d >= 0; d = d - 1) {{
                p_contig_strides[d] = accum;
                accum = accum * layouts[0u].shape[d];
            }}

            for (var d: u32 = 0u; d < layouts[0u].ndim; d = d + 1u) {{
                let coord = temp / p_contig_strides[d];
                temp = temp % p_contig_strides[d];

                off_p = off_p + coord * layouts[0u].strides[d];

                if (d >= layouts[0u].ndim - layouts[1u].ndim) {{
                    let gd = d + layouts[1u].ndim - layouts[0u].ndim;
                    if (layouts[1u].shape[gd] > 1u) {{
                        off_g = off_g + coord * layouts[1u].strides[gd];
                    }}
                }}
                if (d >= layouts[0u].ndim - layouts[2u].ndim) {{
                    let md = d + layouts[2u].ndim - layouts[0u].ndim;
                    if (layouts[2u].shape[md] > 1u) {{
                        off_m = off_m + coord * layouts[2u].strides[md];
                    }}
                }}
                if (d >= layouts[0u].ndim - layouts[3u].ndim) {{
                    let vd = d + layouts[3u].ndim - layouts[0u].ndim;
                    if (layouts[3u].shape[vd] > 1u) {{
                        off_v = off_v + coord * layouts[3u].strides[vd];
                    }}
                }}
            }}

            if (off_p >= arrayLength(&param) || off_g >= arrayLength(&grad) || off_m >= arrayLength(&m) || off_v >= arrayLength(&v)) {{
                return;
            }}

            let g = grad[off_g];
            let m_val = m[off_m] * params.beta1 + ({one_val} - params.beta1) * g;
            let v_val = v[off_v] * params.beta2 + ({one_val} - params.beta2) * g * g;

            m[off_m] = m_val;
            v[off_v] = v_val;

            let m_hat = m_val / params.bias_correction1;
            let v_hat = v_val / params.bias_correction2;
            let denom = sqrt(v_hat) + params.eps;
            param[off_p] = param[off_p] * ({one_val} - params.lr * params.weight_decay) - params.lr * m_hat / denom;
        }}
        "#,
        wgsl_type = wgsl_type,
        one_val = T::WGSL_ONE,
    );

    let key = format!("adamw_step_{}", wgsl_type);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("adamw-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grad.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: m.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: layouts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adamw-encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("adamw-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = len.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}
