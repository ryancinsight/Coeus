#![allow(clippy::too_many_arguments)]

use coeus_core::Layout;
use crate::backend::WgpuScalar;

use crate::kernels::cache::PIPELINE_CACHE;
use crate::kernels::layout::GpuLayoutInfo;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SgdParams<T> {
    pub lr: T,
    pub momentum: T,
}
unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for SgdParams<T> {}
unsafe impl<T: bytemuck::Pod> bytemuck::Pod for SgdParams<T> {}

pub fn dispatch_sgd_step<T: WgpuScalar + coeus_core::Float>(
    param: &wgpu::Buffer,
    param_layout: &Layout,
    grad: &wgpu::Buffer,
    grad_layout: &Layout,
    velocity: &wgpu::Buffer,
    velocity_layout: &Layout,
    lr: T,
    momentum: T,
    len: usize,
) {
    let ctx = crate::backend::get_wgpu_context();

    let p_layout_gpu = GpuLayoutInfo::from_layout(param_layout);
    let g_layout_gpu = GpuLayoutInfo::from_layout(grad_layout);
    let v_layout_gpu = GpuLayoutInfo::from_layout(velocity_layout);

    let params_data = SgdParams { lr, momentum };

    let p_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let g_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let v_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let params_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue.write_buffer(&p_layout_buf, 0, bytemuck::bytes_of(&p_layout_gpu));
    ctx.queue.write_buffer(&g_layout_buf, 0, bytemuck::bytes_of(&g_layout_gpu));
    ctx.queue.write_buffer(&v_layout_buf, 0, bytemuck::bytes_of(&v_layout_gpu));
    ctx.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params_data));

    let wgsl_type = T::WGSL_TYPE;

    let shader_src = format!(
        r#"
        struct LayoutInfo {{
            offset: u32,
            ndim: u32,
            shape: array<u32, 8>,
            strides: array<u32, 8>,
        }}

        struct SgdParams {{
            lr: {wgsl_type},
            momentum: {wgsl_type},
        }}

        @group(0) @binding(0) var<storage, read_write> param: array<{wgsl_type}>;
        @group(0) @binding(1) var<storage, read> grad: array<{wgsl_type}>;
        @group(0) @binding(2) var<storage, read_write> velocity: array<{wgsl_type}>;
        @group(0) @binding(3) var<storage, read> param_layout: LayoutInfo;
        @group(0) @binding(4) var<storage, read> grad_layout: LayoutInfo;
        @group(0) @binding(5) var<storage, read> velocity_layout: LayoutInfo;
        @group(0) @binding(6) var<storage, read> params: SgdParams;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let idx = global_id.x;
            if (idx >= arrayLength(&param)) {{
                return;
            }}
            
            var temp = idx;
            var off_p = param_layout.offset;
            var off_g = grad_layout.offset;
            var off_v = velocity_layout.offset;

            var p_contig_strides: array<u32, 8>;
            var accum: u32 = 1u;
            for (var d: i32 = i32(param_layout.ndim) - 1; d >= 0; d = d - 1) {{
                p_contig_strides[d] = accum;
                accum = accum * param_layout.shape[d];
            }}

            for (var d: u32 = 0u; d < param_layout.ndim; d = d + 1u) {{
                let coord = temp / p_contig_strides[d];
                temp = temp % p_contig_strides[d];

                off_p = off_p + coord * param_layout.strides[d];

                if (d >= param_layout.ndim - grad_layout.ndim) {{
                    let gd = d + grad_layout.ndim - param_layout.ndim;
                    if (grad_layout.shape[gd] > 1u) {{
                        off_g = off_g + coord * grad_layout.strides[gd];
                    }}
                }}
                if (d >= param_layout.ndim - velocity_layout.ndim) {{
                    let vd = d + velocity_layout.ndim - param_layout.ndim;
                    if (velocity_layout.shape[vd] > 1u) {{
                        off_v = off_v + coord * velocity_layout.strides[vd];
                    }}
                }}
            }}

            if (off_p >= arrayLength(&param) || off_g >= arrayLength(&grad) || off_v >= arrayLength(&velocity)) {{
                return;
            }}

            let g = grad[off_g];
            let v = velocity[off_v] * params.momentum + g;
            velocity[off_v] = v;
            param[off_p] = param[off_p] - params.lr * v;
        }}
        "#,
        wgsl_type = wgsl_type
    );

    let key = format!("sgd_step_{}", wgsl_type);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sgd-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: param.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: grad.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: velocity.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: p_layout_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: g_layout_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: v_layout_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sgd-encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sgd-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = len.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(std::iter::once(encoder.finish()));
}
