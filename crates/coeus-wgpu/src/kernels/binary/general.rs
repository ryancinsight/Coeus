use super::super::cache::PIPELINE_CACHE;
use super::super::layout::GpuLayoutInfo;
use crate::backend::{checked_workgroup_count, WgpuBackendError, WgpuScalar};

fn wgsl_cmp_literals(wgsl_type: &str) -> (&'static str, &'static str) {
    match wgsl_type {
        "f32" | "f16" => ("0.0", "1.0"),
        "i32" => ("0", "1"),
        "u32" => ("0u", "1u"),
        _ => ("0", "1"),
    }
}

fn wgsl_rhs_expr(op: coeus_ops::BinaryOp, wgsl_type: &str, a: &str, b: &str) -> String {
    use coeus_ops::BinaryOp;
    let (z, o) = wgsl_cmp_literals(wgsl_type);
    match op {
        BinaryOp::Add => format!("{a} + {b}"),
        BinaryOp::Sub => format!("{a} - {b}"),
        BinaryOp::Mul => format!("{a} * {b}"),
        BinaryOp::Div => format!("{a} / {b}"),
        BinaryOp::Eq => format!("select({z}, {o}, {a} == {b})"),
        BinaryOp::Ne => format!("select({z}, {o}, {a} != {b})"),
        BinaryOp::Lt => format!("select({z}, {o}, {a} < {b})"),
        BinaryOp::Gt => format!("select({z}, {o}, {a} > {b})"),
        BinaryOp::Le => format!("select({z}, {o}, {a} <= {b})"),
        BinaryOp::Ge => format!("select({z}, {o}, {a} >= {b})"),
    }
}

/// Dispatch a WGSL shader for general elementwise binary operations with layout traversal and broadcasting.
#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub fn dispatch_binary<T: WgpuScalar>(
    op: coeus_ops::BinaryOp,
    a: &wgpu::Buffer,
    a_layout: &coeus_core::Layout,
    b: &wgpu::Buffer,
    b_layout: &coeus_core::Layout,
    c: &wgpu::Buffer,
    c_layout: &coeus_core::Layout,
    len: usize,
) -> Result<(), WgpuBackendError> {
    let workgroups = checked_workgroup_count("binary", len)?;
    let a_layout_gpu = GpuLayoutInfo::try_from_layout(a_layout)
        .map_err(|error| WgpuBackendError::Layout(error.into()))?;
    let b_layout_gpu = GpuLayoutInfo::try_from_layout(b_layout)
        .map_err(|error| WgpuBackendError::Layout(error.into()))?;
    let c_layout_gpu = GpuLayoutInfo::try_from_layout(c_layout)
        .map_err(|error| WgpuBackendError::Layout(error.into()))?;
    let ctx = crate::backend::get_wgpu_context();
    let wgsl_type = T::WGSL_TYPE;

    let a_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let b_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let c_layout_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue
        .write_buffer(&a_layout_buf, 0, bytemuck::bytes_of(&a_layout_gpu));
    ctx.queue
        .write_buffer(&b_layout_buf, 0, bytemuck::bytes_of(&b_layout_gpu));
    ctx.queue
        .write_buffer(&c_layout_buf, 0, bytemuck::bytes_of(&c_layout_gpu));

    let is_a_c = std::ptr::eq(a, c);
    let is_b_c = std::ptr::eq(b, c);

    let (a_ref, b_ref) = if is_a_c && is_b_c {
        ("c[off_a]", "c[off_b]")
    } else if is_a_c {
        ("c[off_a]", "b[off_b]")
    } else if is_b_c {
        ("a[off_a]", "c[off_b]")
    } else {
        ("a[off_a]", "b[off_b]")
    };
    let rhs = wgsl_rhs_expr(op, wgsl_type, a_ref, b_ref);

    let key = format!("binary_{:?}_{}_ac_{}_bc_{}", op, wgsl_type, is_a_c, is_b_c);

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

                c[idx] = {};
            }}
            "#,
            wgsl_type, rhs
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

            @group(0) @binding(0) var<storage, read> b: array<{0}>;
            @group(0) @binding(1) var<storage, read_write> c: array<{0}>;
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

                c[idx] = {1};
            }}
            "#,
            wgsl_type, rhs
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

            @group(0) @binding(0) var<storage, read> a: array<{0}>;
            @group(0) @binding(1) var<storage, read_write> c: array<{0}>;
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

                c[idx] = {1};
            }}
            "#,
            wgsl_type, rhs
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

            @group(0) @binding(0) var<storage, read> a: array<{0}>;
            @group(0) @binding(1) var<storage, read> b: array<{0}>;
            @group(0) @binding(2) var<storage, read_write> c: array<{0}>;
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

                c[idx] = {1};
            }}
            "#,
            wgsl_type, rhs
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
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    Ok(())
}
