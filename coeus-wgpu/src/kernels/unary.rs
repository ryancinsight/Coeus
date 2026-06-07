use super::cache::PIPELINE_CACHE;
use super::layout::GpuLayoutInfo;
use crate::backend::WgpuScalar;

/// Dispatch a WGSL shader for general elementwise unary operations with layout traversal.
pub fn dispatch_unary<T: WgpuScalar>(
    op: coeus_ops::UnaryOp,
    a: &wgpu::Buffer,
    a_layout: &coeus_core::Layout,
    c: &wgpu::Buffer,
    c_layout: &coeus_core::Layout,
    len: usize,
) {
    let ctx = crate::backend::get_wgpu_context();
    let wgsl_type = T::WGSL_TYPE;

    let a_layout_gpu = GpuLayoutInfo::from_layout(a_layout);
    let c_layout_gpu = GpuLayoutInfo::from_layout(c_layout);

    let a_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let c_layout_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue.write_buffer(&a_layout_buf, 0, bytemuck::bytes_of(&a_layout_gpu));
    ctx.queue.write_buffer(&c_layout_buf, 0, bytemuck::bytes_of(&c_layout_gpu));

    let expr: String = match op {
        coeus_ops::UnaryOp::Relu => "max(val, 0.0)".to_string(),
        coeus_ops::UnaryOp::ReluGrad => "select(0.0, 1.0, val > 0.0)".to_string(),
        coeus_ops::UnaryOp::Sigmoid => "1.0 / (1.0 + exp(-val))".to_string(),
        coeus_ops::UnaryOp::SigmoidGrad => "val * (1.0 - val)".to_string(),
        coeus_ops::UnaryOp::Tanh => "tanh(val)".to_string(),
        coeus_ops::UnaryOp::TanhGrad => "1.0 - val * val".to_string(),
        coeus_ops::UnaryOp::Gelu => "0.5 * val * (1.0 + tanh(0.7978845608 * (val + 0.044715 * val * val * val)))".to_string(),
        coeus_ops::UnaryOp::GeluGrad => "0.5 * (1.0 + tanh(0.7978845608 * (val + 0.044715 * val * val * val))) + 0.5 * val * (1.0 - tanh(0.7978845608 * (val + 0.044715 * val * val * val)) * tanh(0.7978845608 * (val + 0.044715 * val * val * val))) * 0.7978845608 * (1.0 + 0.134145 * val * val)".to_string(),
        coeus_ops::UnaryOp::Sin => "sin(val)".to_string(),
        coeus_ops::UnaryOp::Cos => "cos(val)".to_string(),
        coeus_ops::UnaryOp::Exp => "exp(val)".to_string(),
        coeus_ops::UnaryOp::Log => "log(val)".to_string(),
        coeus_ops::UnaryOp::Neg => "-val".to_string(),
        coeus_ops::UnaryOp::Abs => "abs(val)".to_string(),
        coeus_ops::UnaryOp::Sqrt => "sqrt(val)".to_string(),
        coeus_ops::UnaryOp::Silu => "val / (1.0 + exp(-val))".to_string(),
        coeus_ops::UnaryOp::SiluGrad => "(1.0 / (1.0 + exp(-val))) * (1.0 + val * (1.0 - (1.0 / (1.0 + exp(-val)))))".to_string(),
        coeus_ops::UnaryOp::Mish => "val * tanh(log(1.0 + exp(val)))".to_string(),
        coeus_ops::UnaryOp::MishGrad => "tanh(log(1.0 + exp(val))) + val * (1.0 - tanh(log(1.0 + exp(val))) * tanh(log(1.0 + exp(val)))) * (1.0 / (1.0 + exp(-val)))".to_string(),
        coeus_ops::UnaryOp::Elu => "select(exp(val) - 1.0, val, val >= 0.0)".to_string(),
        coeus_ops::UnaryOp::EluGrad => "select(exp(val), 1.0, val >= 0.0)".to_string(),
        coeus_ops::UnaryOp::Softplus => "log(1.0 + exp(val))".to_string(),
        coeus_ops::UnaryOp::SoftplusGrad => "1.0 / (1.0 + exp(-val))".to_string(),
        coeus_ops::UnaryOp::GeluTanh => "0.5 * val * (1.0 + tanh(0.7978845608 * (val + 0.044715 * val * val * val)))".to_string(),
        coeus_ops::UnaryOp::GeluTanhGrad => "0.5 * (1.0 + tanh(0.7978845608 * (val + 0.044715 * val * val * val))) + 0.5 * val * (1.0 - tanh(0.7978845608 * (val + 0.044715 * val * val * val)) * tanh(0.7978845608 * (val + 0.044715 * val * val * val))) * 0.7978845608 * (1.0 + 0.134145 * val * val)".to_string(),
        coeus_ops::UnaryOp::LeakyRelu(slope_bits) => {
            let slope = f64::from_bits(slope_bits);
            format!("select({slope:.17} * val, val, val >= 0.0)")
        },
        coeus_ops::UnaryOp::LeakyReluGrad(slope_bits) => {
            let slope = f64::from_bits(slope_bits);
            format!("select({slope:.17}, 1.0, val >= 0.0)")
        },
    };

    let is_inplace = std::ptr::eq(a, c);
    let key = format!("unary_{:?}_{}_inplace_{}", op, wgsl_type, is_inplace);

    let shader_src = if is_inplace {
        format!(
            r#"
            struct LayoutInfo {{
                offset: u32,
                ndim: u32,
                shape: array<u32, 8>,
                strides: array<u32, 8>,
            }}

            @group(0) @binding(0) var<storage, read_write> c: array<{}>;
            @group(0) @binding(1) var<storage, read> a_layout: LayoutInfo;
            @group(0) @binding(2) var<storage, read> c_layout: LayoutInfo;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                
                var temp = idx;
                var off_a = a_layout.offset;

                for (var d: u32 = 0u; d < c_layout.ndim; d = d + 1u) {{
                    let coord = temp / c_layout.strides[d];
                    temp = temp % c_layout.strides[d];

                    if (d >= c_layout.ndim - a_layout.ndim) {{
                        let ad = d + a_layout.ndim - c_layout.ndim;
                        if (a_layout.shape[ad] > 1u) {{
                            off_a = off_a + coord * a_layout.strides[ad];
                        }}
                    }}
                }}

                let val = c[off_a];
                c[idx] = {};
            }}
            "#,
            wgsl_type, expr
        )
    } else {
        format!(
            r#"
            struct LayoutInfo {{
                offset: u32,
                ndim: u32,
                shape: array<u32, 8>,
                strides: array<u32, 8>,
            }}

            @group(0) @binding(0) var<storage, read> a: array<{}>;
            @group(0) @binding(1) var<storage, read_write> c: array<{}>;
            @group(0) @binding(2) var<storage, read> a_layout: LayoutInfo;
            @group(0) @binding(3) var<storage, read> c_layout: LayoutInfo;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                
                var temp = idx;
                var off_a = a_layout.offset;

                for (var d: u32 = 0u; d < c_layout.ndim; d = d + 1u) {{
                    let coord = temp / c_layout.strides[d];
                    temp = temp % c_layout.strides[d];

                    if (d >= c_layout.ndim - a_layout.ndim) {{
                        let ad = d + a_layout.ndim - c_layout.ndim;
                        if (a_layout.shape[ad] > 1u) {{
                            off_a = off_a + coord * a_layout.strides[ad];
                        }}
                    }}
                }}

                let val = a[off_a];
                c[idx] = {};
            }}
            "#,
            wgsl_type, wgsl_type, expr
        )
    };

    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = if is_inplace {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary-bind-group-inplace"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_layout_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_layout_buf.as_entire_binding() },
            ],
        })
    } else {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: a_layout_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_layout_buf.as_entire_binding() },
            ],
        })
    };

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("unary-encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("unary-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = len.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Dispatch a WGSL shader for flat contiguous elementwise unary operations without layout traversal.
pub fn dispatch_contiguous_unary<T: WgpuScalar>(
    op: coeus_ops::UnaryOp,
    a: &wgpu::Buffer,
    c: &wgpu::Buffer,
    len: usize,
) {
    let ctx = crate::backend::get_wgpu_context();
    let wgsl_type = T::WGSL_TYPE;

    let expr: String = match op {
        coeus_ops::UnaryOp::Relu => "max(val, 0.0)".to_string(),
        coeus_ops::UnaryOp::ReluGrad => "select(0.0, 1.0, val > 0.0)".to_string(),
        coeus_ops::UnaryOp::Sigmoid => "1.0 / (1.0 + exp(-val))".to_string(),
        coeus_ops::UnaryOp::SigmoidGrad => "val * (1.0 - val)".to_string(),
        coeus_ops::UnaryOp::Tanh => "tanh(val)".to_string(),
        coeus_ops::UnaryOp::TanhGrad => "1.0 - val * val".to_string(),
        coeus_ops::UnaryOp::Gelu => "0.5 * val * (1.0 + tanh(0.7978845608 * (val + 0.044715 * val * val * val)))".to_string(),
        coeus_ops::UnaryOp::GeluGrad => "0.5 * (1.0 + tanh(0.7978845608 * (val + 0.044715 * val * val * val))) + 0.5 * val * (1.0 - tanh(0.7978845608 * (val + 0.044715 * val * val * val)) * tanh(0.7978845608 * (val + 0.044715 * val * val * val))) * 0.7978845608 * (1.0 + 0.134145 * val * val)".to_string(),
        coeus_ops::UnaryOp::Sin => "sin(val)".to_string(),
        coeus_ops::UnaryOp::Cos => "cos(val)".to_string(),
        coeus_ops::UnaryOp::Exp => "exp(val)".to_string(),
        coeus_ops::UnaryOp::Log => "log(val)".to_string(),
        coeus_ops::UnaryOp::Neg => "-val".to_string(),
        coeus_ops::UnaryOp::Abs => "abs(val)".to_string(),
        coeus_ops::UnaryOp::Sqrt => "sqrt(val)".to_string(),
        coeus_ops::UnaryOp::Silu => "val / (1.0 + exp(-val))".to_string(),
        coeus_ops::UnaryOp::SiluGrad => "(1.0 / (1.0 + exp(-val))) * (1.0 + val * (1.0 - (1.0 / (1.0 + exp(-val)))))".to_string(),
        coeus_ops::UnaryOp::Mish => "val * tanh(log(1.0 + exp(val)))".to_string(),
        coeus_ops::UnaryOp::MishGrad => "tanh(log(1.0 + exp(val))) + val * (1.0 - tanh(log(1.0 + exp(val))) * tanh(log(1.0 + exp(val)))) * (1.0 / (1.0 + exp(-val)))".to_string(),
        coeus_ops::UnaryOp::Elu => "select(exp(val) - 1.0, val, val >= 0.0)".to_string(),
        coeus_ops::UnaryOp::EluGrad => "select(exp(val), 1.0, val >= 0.0)".to_string(),
        coeus_ops::UnaryOp::Softplus => "log(1.0 + exp(val))".to_string(),
        coeus_ops::UnaryOp::SoftplusGrad => "1.0 / (1.0 + exp(-val))".to_string(),
        coeus_ops::UnaryOp::GeluTanh => "0.5 * val * (1.0 + tanh(0.7978845608 * (val + 0.044715 * val * val * val)))".to_string(),
        coeus_ops::UnaryOp::GeluTanhGrad => "0.5 * (1.0 + tanh(0.7978845608 * (val + 0.044715 * val * val * val))) + 0.5 * val * (1.0 - tanh(0.7978845608 * (val + 0.044715 * val * val * val)) * tanh(0.7978845608 * (val + 0.044715 * val * val * val))) * 0.7978845608 * (1.0 + 0.134145 * val * val)".to_string(),
        coeus_ops::UnaryOp::LeakyRelu(slope_bits) => {
            let slope = f64::from_bits(slope_bits);
            format!("select({slope:.17} * val, val, val >= 0.0)")
        },
        coeus_ops::UnaryOp::LeakyReluGrad(slope_bits) => {
            let slope = f64::from_bits(slope_bits);
            format!("select({slope:.17}, 1.0, val >= 0.0)")
        },
    };

    let is_inplace = std::ptr::eq(a, c);
    let key = format!("contiguous_unary_{:?}_{}_inplace_{}", op, wgsl_type, is_inplace);

    let shader_src = if is_inplace {
        format!(
            r#"
            @group(0) @binding(0) var<storage, read_write> c: array<{}>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                let val = c[idx];
                c[idx] = {};
            }}
            "#,
            wgsl_type, expr
        )
    } else {
        format!(
            r#"
            @group(0) @binding(0) var<storage, read> a: array<{}>;
            @group(0) @binding(1) var<storage, read_write> c: array<{}>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                let val = a[idx];
                c[idx] = {};
            }}
            "#,
            wgsl_type, wgsl_type, expr
        )
    };

    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = if is_inplace {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary-bind-group-inplace"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: c.as_entire_binding() },
            ],
        })
    } else {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: c.as_entire_binding() },
            ],
        })
    };

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("unary-encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("unary-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = len.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}
