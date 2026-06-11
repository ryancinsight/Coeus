use super::super::cache::PIPELINE_CACHE;
use super::super::layout::GpuLayoutInfo;
use crate::backend::WgpuScalar;

/// Dispatch a WGSL shader for general elementwise binary operations with layout traversal and broadcasting.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_binary<T: WgpuScalar>(
    op: coeus_ops::BinaryOp,
    a: &wgpu::Buffer,
    a_layout: &coeus_core::Layout,
    b: &wgpu::Buffer,
    b_layout: &coeus_core::Layout,
    c: &wgpu::Buffer,
    c_layout: &coeus_core::Layout,
    len: usize,
) {
    let ctx = crate::backend::get_wgpu_context();
    let wgsl_type = T::WGSL_TYPE;

    let a_layout_gpu = GpuLayoutInfo::from_layout(a_layout);
    let b_layout_gpu = GpuLayoutInfo::from_layout(b_layout);
    let c_layout_gpu = GpuLayoutInfo::from_layout(c_layout);

    let a_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let b_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let c_layout_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue
        .write_buffer(&a_layout_buf, 0, bytemuck::bytes_of(&a_layout_gpu));
    ctx.queue
        .write_buffer(&b_layout_buf, 0, bytemuck::bytes_of(&b_layout_gpu));
    ctx.queue
        .write_buffer(&c_layout_buf, 0, bytemuck::bytes_of(&c_layout_gpu));

    let op_symbol = match op {
        coeus_ops::BinaryOp::Add => "+",
        coeus_ops::BinaryOp::Sub => "-",
        coeus_ops::BinaryOp::Mul => "*",
        coeus_ops::BinaryOp::Div => "/",
    };

    let is_a_c = std::ptr::eq(a, c);
    let is_b_c = std::ptr::eq(b, c);

    let key = format!(
        "binary_{}_{}_ac_{}_bc_{}",
        op_symbol, wgsl_type, is_a_c, is_b_c
    );

    let shader_src = if is_a_c && is_b_c {
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
            @group(0) @binding(2) var<storage, read> b_layout: LayoutInfo;
            @group(0) @binding(3) var<storage, read> c_layout: LayoutInfo;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                
                var temp = idx;
                var off_a = a_layout.offset;
                var off_b = b_layout.offset;

                for (var d: u32 = 0u; d < c_layout.ndim; d = d + 1u) {{
                    let coord = temp / c_layout.strides[d];
                    temp = temp % c_layout.strides[d];

                    if (d >= c_layout.ndim - a_layout.ndim) {{
                        let ad = d + a_layout.ndim - c_layout.ndim;
                        if (a_layout.shape[ad] > 1u) {{
                            off_a = off_a + coord * a_layout.strides[ad];
                        }}
                    }}
                    if (d >= c_layout.ndim - b_layout.ndim) {{
                        let bd = d + b_layout.ndim - c_layout.ndim;
                        if (b_layout.shape[bd] > 1u) {{
                            off_b = off_b + coord * b_layout.strides[bd];
                        }}
                    }}
                }}

                c[idx] = c[off_a] {} c[off_b];
            }}
            "#,
            wgsl_type, op_symbol
        )
    } else if is_a_c {
        format!(
            r#"
            struct LayoutInfo {{
                offset: u32,
                ndim: u32,
                shape: array<u32, 8>,
                strides: array<u32, 8>,
            }}

            @group(0) @binding(0) var<storage, read> b: array<{}>;
            @group(0) @binding(1) var<storage, read_write> c: array<{}>;
            @group(0) @binding(2) var<storage, read> a_layout: LayoutInfo;
            @group(0) @binding(3) var<storage, read> b_layout: LayoutInfo;
            @group(0) @binding(4) var<storage, read> c_layout: LayoutInfo;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                
                var temp = idx;
                var off_a = a_layout.offset;
                var off_b = b_layout.offset;

                for (var d: u32 = 0u; d < c_layout.ndim; d = d + 1u) {{
                    let coord = temp / c_layout.strides[d];
                    temp = temp % c_layout.strides[d];

                    if (d >= c_layout.ndim - a_layout.ndim) {{
                        let ad = d + a_layout.ndim - c_layout.ndim;
                        if (a_layout.shape[ad] > 1u) {{
                            off_a = off_a + coord * a_layout.strides[ad];
                        }}
                    }}
                    if (d >= c_layout.ndim - b_layout.ndim) {{
                        let bd = d + b_layout.ndim - c_layout.ndim;
                        if (b_layout.shape[bd] > 1u) {{
                            off_b = off_b + coord * b_layout.strides[bd];
                        }}
                    }}
                }}

                c[idx] = c[off_a] {} b[off_b];
            }}
            "#,
            wgsl_type, wgsl_type, op_symbol
        )
    } else if is_b_c {
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
            @group(0) @binding(3) var<storage, read> b_layout: LayoutInfo;
            @group(0) @binding(4) var<storage, read> c_layout: LayoutInfo;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                
                var temp = idx;
                var off_a = a_layout.offset;
                var off_b = b_layout.offset;

                for (var d: u32 = 0u; d < c_layout.ndim; d = d + 1u) {{
                    let coord = temp / c_layout.strides[d];
                    temp = temp % c_layout.strides[d];

                    if (d >= c_layout.ndim - a_layout.ndim) {{
                        let ad = d + a_layout.ndim - c_layout.ndim;
                        if (a_layout.shape[ad] > 1u) {{
                            off_a = off_a + coord * a_layout.strides[ad];
                        }}
                    }}
                    if (d >= c_layout.ndim - b_layout.ndim) {{
                        let bd = d + b_layout.ndim - c_layout.ndim;
                        if (b_layout.shape[bd] > 1u) {{
                            off_b = off_b + coord * b_layout.strides[bd];
                        }}
                    }}
                }}

                c[idx] = a[off_a] {} c[off_b];
            }}
            "#,
            wgsl_type, wgsl_type, op_symbol
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
            @group(0) @binding(1) var<storage, read> b: array<{}>;
            @group(0) @binding(2) var<storage, read_write> c: array<{}>;
            @group(0) @binding(3) var<storage, read> a_layout: LayoutInfo;
            @group(0) @binding(4) var<storage, read> b_layout: LayoutInfo;
            @group(0) @binding(5) var<storage, read> c_layout: LayoutInfo;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                
                var temp = idx;
                var off_a = a_layout.offset;
                var off_b = b_layout.offset;

                for (var d: u32 = 0u; d < c_layout.ndim; d = d + 1u) {{
                    let coord = temp / c_layout.strides[d];
                    temp = temp % c_layout.strides[d];

                    if (d >= c_layout.ndim - a_layout.ndim) {{
                        let ad = d + a_layout.ndim - c_layout.ndim;
                        if (a_layout.shape[ad] > 1u) {{
                            off_a = off_a + coord * a_layout.strides[ad];
                        }}
                    }}
                    if (d >= c_layout.ndim - b_layout.ndim) {{
                        let bd = d + b_layout.ndim - c_layout.ndim;
                        if (b_layout.shape[bd] > 1u) {{
                            off_b = off_b + coord * b_layout.strides[bd];
                        }}
                    }}
                }}

                c[idx] = a[off_a] {} b[off_b];
            }}
            "#,
            wgsl_type, wgsl_type, wgsl_type, op_symbol
        )
    };

    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = if is_a_c && is_b_c {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary-bind-group-abc"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c_layout_buf.as_entire_binding(),
                },
            ],
        })
    } else if is_a_c {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary-bind-group-ac"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: c_layout_buf.as_entire_binding(),
                },
            ],
        })
    } else if is_b_c {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary-bind-group-bc"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: c_layout_buf.as_entire_binding(),
                },
            ],
        })
    } else {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: a_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: c_layout_buf.as_entire_binding(),
                },
            ],
        })
    };

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("binary-encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("binary-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = len.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}
