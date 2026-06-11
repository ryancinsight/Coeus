use super::super::cache::PIPELINE_CACHE;
use crate::backend::WgpuScalar;

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

    let op_symbol = match op {
        coeus_ops::BinaryOp::Add => "+",
        coeus_ops::BinaryOp::Sub => "-",
        coeus_ops::BinaryOp::Mul => "*",
        coeus_ops::BinaryOp::Div => "/",
    };

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
                c[idx] = c[idx] {} c[idx];
            }}
            "#,
            wgsl_type, op_symbol
        )
    } else if is_a_c {
        format!(
            r#"
            @group(0) @binding(0) var<storage, read> b: array<{}>;
            @group(0) @binding(1) var<storage, read_write> c: array<{}>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                c[idx] = c[idx] {} b[idx];
            }}
            "#,
            wgsl_type, wgsl_type, op_symbol
        )
    } else if is_b_c {
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
                c[idx] = a[idx] {} c[idx];
            }}
            "#,
            wgsl_type, wgsl_type, op_symbol
        )
    } else {
        format!(
            r#"
            @group(0) @binding(0) var<storage, read> a: array<{}>;
            @group(0) @binding(1) var<storage, read> b: array<{}>;
            @group(0) @binding(2) var<storage, read_write> c: array<{}>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= arrayLength(&c)) {{
                    return;
                }}
                c[idx] = a[idx] {} b[idx];
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
