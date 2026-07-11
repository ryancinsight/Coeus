use super::super::cache::PIPELINE_CACHE;
use crate::backend::WgpuScalar;

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

/// Dispatch a WGSL shader for flat contiguous elementwise binary operations: c = a op b.
pub fn dispatch_contiguous_binary<T: WgpuScalar>(
    op: coeus_ops::BinaryOp,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    c: &wgpu::Buffer,
    len: usize,
) {
    let ctx = crate::backend::get_wgpu_context();
    let wgsl_type = T::WGSL_TYPE;

    let is_a_c = std::ptr::eq(a, c);
    let is_b_c = std::ptr::eq(b, c);

    let (a_ref, b_ref) = if is_a_c && is_b_c {
        ("c[idx]", "c[idx]")
    } else if is_a_c {
        ("c[idx]", "b[idx]")
    } else if is_b_c {
        ("a[idx]", "c[idx]")
    } else {
        ("a[idx]", "b[idx]")
    };
    let rhs = wgsl_rhs_expr(op, wgsl_type, a_ref, b_ref);

    let key = format!(
        "contiguous_binary_{:?}_{}_ac_{}_bc_{}",
        op, wgsl_type, is_a_c, is_b_c
    );

    let shader_src = if is_a_c && is_b_c {
        format!(
            r#"
            @group(0) @binding(0) var<storage, read_write> c: array<{}>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                c[idx] = {1};
            }}
            "#,
            wgsl_type, rhs
        )
    } else if is_a_c {
        format!(
            r#"
            @group(0) @binding(0) var<storage, read> b: array<{0}>;
            @group(0) @binding(1) var<storage, read_write> c: array<{0}>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                c[idx] = {1};
            }}
            "#,
            wgsl_type, rhs
        )
    } else if is_b_c {
        format!(
            r#"
            @group(0) @binding(0) var<storage, read> a: array<{0}>;
            @group(0) @binding(1) var<storage, read_write> c: array<{0}>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                c[idx] = {1};
            }}
            "#,
            wgsl_type, rhs
        )
    } else {
        format!(
            r#"
            @group(0) @binding(0) var<storage, read> a: array<{0}>;
            @group(0) @binding(1) var<storage, read> b: array<{0}>;
            @group(0) @binding(2) var<storage, read_write> c: array<{0}>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
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
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: c.as_entire_binding(),
            }],
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
