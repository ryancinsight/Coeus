use super::cache::PIPELINE_CACHE;
use super::layout::GpuLayoutInfo;
use crate::backend::{WgpuBackend, WgpuScalar};
use crate::storage::WgpuStorage;
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;
use std::collections::HashMap;

/// Dispatch a WGSL shader for reduction operations along an axis.
pub fn dispatch_reduce<T: WgpuScalar>(
    op: coeus_ops::ReductionOp,
    a: &wgpu::Buffer,
    a_layout: &coeus_core::Layout,
    axis: usize,
    c: &wgpu::Buffer,
    c_layout: &coeus_core::Layout,
) {
    let ctx = crate::backend::get_wgpu_context();
    let wgsl_type = T::WGSL_TYPE;

    let a_layout_gpu = GpuLayoutInfo::from_layout(a_layout);
    let c_layout_gpu = GpuLayoutInfo::from_layout(c_layout);
    let axis_gpu = axis as u32;

    let a_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let c_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let axis_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue
        .write_buffer(&a_layout_buf, 0, bytemuck::bytes_of(&a_layout_gpu));
    ctx.queue
        .write_buffer(&c_layout_buf, 0, bytemuck::bytes_of(&c_layout_gpu));
    ctx.queue
        .write_buffer(&axis_buf, 0, bytemuck::bytes_of(&axis_gpu));

    let (init_expr, loop_start, update_expr) = match op {
        coeus_ops::ReductionOp::Sum => ("0.0", "0u", "acc = acc + val;"),
        coeus_ops::ReductionOp::Max => ("a[base_off_a]", "1u", "acc = max(acc, val);"),
        coeus_ops::ReductionOp::Min => ("a[base_off_a]", "1u", "acc = min(acc, val);"),
    };

    let shader_src = format!(
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
        @group(0) @binding(4) var<storage, read> axis_info: u32;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let idx = global_id.x;
            if (idx >= arrayLength(&c)) {{
                return;
            }}
            
            let axis = axis_info;
            let axis_len = a_layout.shape[axis];
            
            var temp = idx;
            var base_off_a = a_layout.offset;

            for (var d: u32 = 0u; d < a_layout.ndim; d = d + 1u) {{
                let coord = temp / c_layout.strides[d];
                temp = temp % c_layout.strides[d];
                if (d != axis) {{
                    base_off_a = base_off_a + coord * a_layout.strides[d];
                }}
            }}

            var acc = {};
            let stride_axis = a_layout.strides[axis];
            if (axis_len > 0u) {{
                for (var k: u32 = {}; k < axis_len; k = k + 1u) {{
                    let val = a[base_off_a + k * stride_axis];
                    {}
                }}
            }}
            
            c[idx] = acc;
        }}
        "#,
        wgsl_type, wgsl_type, init_expr, loop_start, update_expr
    );

    let key = format!("reduce_{:?}_{}", op, wgsl_type);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reduce-bind-group"),
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
                resource: c_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: axis_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reduce-encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reduce-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let out_numel = c_layout.shape().iter().product::<usize>();
        let workgroups = out_numel.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Dispatch a WGSL shader for fused element-wise and reduction along an axis.
pub fn dispatch_fused_reduce<T: WgpuScalar, E: ExprNode<T, WgpuBackend>>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
    c: &mut WgpuStorage<T>,
    c_layout: &coeus_core::Layout,
) {
    let ctx = crate::backend::get_wgpu_context();
    let wgsl_type = T::WGSL_TYPE;

    let expr_shape = expr
        .shape()
        .expect("Fused expression must have at least one tensor input");
    let expr_ndim = expr_shape.len() as u32;
    let axis_len_gpu = expr_shape[axis] as u32;
    let axis_gpu = axis as u32;

    // 1. Collect unique input tensors
    let mut input_ptrs = Vec::new();
    expr.collect_inputs(&mut input_ptrs);
    let num_inputs = input_ptrs.len();

    let inputs: Vec<&Tensor<T, WgpuBackend>> = input_ptrs.iter().map(|&p| unsafe { &*p }).collect();

    // 2. Build input pointer to index map
    let mut input_map = HashMap::new();
    for (i, &p) in input_ptrs.iter().enumerate() {
        input_map.insert(p, i);
    }

    // 3. Generate the shader expression string
    let expr_str = expr.to_shader_expr(&input_map);

    // 4. Create the unified layout buffer
    let mut layouts_gpu = Vec::with_capacity(num_inputs + 1);
    for input in &inputs {
        layouts_gpu.push(GpuLayoutInfo::from_layout(input.layout()));
    }
    // We add c_layout as the last one to decode output coordinates
    layouts_gpu.push(GpuLayoutInfo::from_layout(c_layout));

    let layout_buf = crate::backend::PooledMetadataBuffer::new();
    ctx.queue
        .write_buffer(&layout_buf, 0, bytemuck::cast_slice(&layouts_gpu));

    let axis_info = [axis_gpu, axis_len_gpu];
    let axis_buf = crate::backend::PooledMetadataBuffer::new();
    ctx.queue
        .write_buffer(&axis_buf, 0, bytemuck::cast_slice(&axis_info));

    // 5. Generate the WGSL code
    let mut inputs_decl = String::new();
    let mut offset_calcs = String::new();

    for i in 0..num_inputs {
        inputs_decl.push_str(&format!(
            "        @group(0) @binding({}) var<storage, read> t_{}: array<{}>;\n",
            i, i, wgsl_type
        ));
        offset_calcs.push_str(&format!(
            "            var off_{} = layout_infos[{}].offset;\n\
            for (var d: u32 = 0u; d < {}; d = d + 1u) {{\n\
                if (d >= {} - layout_infos[{}].ndim) {{\n\
                    let ad = d + layout_infos[{}].ndim - {};\n\
                    if (layout_infos[{}].shape[ad] > 1u) {{\n\
                        off_{} = off_{} + coords[d] * layout_infos[{}].strides[ad];\n\
                    }}\n\
                }}\n\
            }}\n\
            let val_{} = t_{}[off_{}];\n\n",
            i, i, expr_ndim, expr_ndim, i, i, expr_ndim, i, i, i, i, i, i, i
        ));
    }

    let (init_expr, update_expr) = match op {
        coeus_ops::ReductionOp::Sum => ("0.0", "acc = acc + val;"),
        coeus_ops::ReductionOp::Max => ("-3.40282347e+38", "acc = max(acc, val);"),
        coeus_ops::ReductionOp::Min => ("3.40282347e+38", "acc = min(acc, val);"),
    };

    let shader_src = format!(
        r#"
        struct LayoutInfo {{
            offset: u32,
            ndim: u32,
            shape: array<u32, 8>,
            strides: array<u32, 8>,
        }}

        struct AxisInfo {{
            axis: u32,
            axis_len: u32,
        }}

{inputs_decl}
        @group(0) @binding({binding_out}) var<storage, read_write> out: array<{wgsl_type}>;
        @group(0) @binding({binding_layouts}) var<storage, read> layout_infos: array<LayoutInfo>;
        @group(0) @binding({binding_axis}) var<storage, read> axis_info: AxisInfo;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let idx = global_id.x;
            if (idx >= arrayLength(&out)) {{
                return;
            }}
            
            let c_layout = layout_infos[{binding_c}];
            let axis = axis_info.axis;
            let axis_len = axis_info.axis_len;
            
            var temp = idx;
            var coords = array<u32, 8>();
            for (var d: u32 = 0u; d < c_layout.ndim; d = d + 1u) {{
                coords[d] = temp / c_layout.strides[d];
                temp = temp % c_layout.strides[d];
            }}

            var acc = {init_expr};
            if (axis_len > 0u) {{
                for (var k: u32 = 0u; k < axis_len; k = k + 1u) {{
                    coords[axis] = k;
{offset_calcs}
                    let val = {expr_str};
                    {update_expr}
                }}
            }}
            
            out[idx] = acc;
        }}
        "#,
        inputs_decl = inputs_decl,
        wgsl_type = wgsl_type,
        binding_out = num_inputs,
        binding_layouts = num_inputs + 1,
        binding_axis = num_inputs + 2,
        binding_c = num_inputs,
        init_expr = init_expr,
        offset_calcs = offset_calcs,
        expr_str = expr_str,
        update_expr = update_expr
    );

    // 6. Create cache key
    let key = format!("fused_reduce_{:?}_{}_{}", op, expr_str, wgsl_type);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");

    // 7. Bind Group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let mut entries = Vec::with_capacity(num_inputs + 3);
    for (i, input) in inputs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32,
            resource: input.storage().buffer.raw().as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: num_inputs as u32,
        resource: c.buffer.raw().as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: (num_inputs + 1) as u32,
        resource: layout_buf.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: (num_inputs + 2) as u32,
        resource: axis_buf.as_entire_binding(),
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fused-reduce-bind-group"),
        layout: &bind_group_layout,
        entries: &entries,
    });

    // 8. Dispatch
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused-reduce-encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused-reduce-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let out_numel = c_layout.shape().iter().product::<usize>();
        let workgroups = out_numel.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}
