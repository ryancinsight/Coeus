use super::cache::PIPELINE_CACHE;
use super::layout::GpuLayoutInfo;
use crate::backend::{WgpuBackend, WgpuScalar};
use crate::storage::WgpuStorage;
use coeus_core::Layout;
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;
use std::collections::HashMap;

/// Compile and dispatch a dynamically generated fused WGSL compute shader on the GPU.
pub fn dispatch_fused<T: WgpuScalar, E: ExprNode<T, WgpuBackend>>(
    expr: &E,
    output: &mut WgpuStorage<T>,
    out_layout: &Layout,
) {
    let ctx = crate::backend::get_wgpu_context();
    let wgsl_type = T::WGSL_TYPE;

    // 1. Collect unique input tensors
    let mut input_ptrs = Vec::new();
    expr.collect_inputs(&mut input_ptrs);
    let num_inputs = input_ptrs.len();

    // Convert raw pointers back to safe references
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
    layouts_gpu.push(GpuLayoutInfo::from_layout(out_layout));

    let layout_buf = crate::backend::PooledMetadataBuffer::new();
    ctx.queue
        .write_buffer(&layout_buf, 0, bytemuck::cast_slice(&layouts_gpu));

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
            for (var d: u32 = 0u; d < out_layout.ndim; d = d + 1u) {{\n\
                if (d >= out_layout.ndim - layout_infos[{}].ndim) {{\n\
                    let ad = d + layout_infos[{}].ndim - out_layout.ndim;\n\
                    if (layout_infos[{}].shape[ad] > 1u) {{\n\
                        off_{} = off_{} + coords[d] * layout_infos[{}].strides[ad];\n\
                    }}\n\
                }}\n\
            }}\n\
            let val_{} = t_{}[off_{}];\n\n",
            i, i, i, i, i, i, i, i, i, i, i
        ));
    }

    let shader_src = format!(
        r#"
        struct LayoutInfo {{
            offset: u32,
            ndim: u32,
            shape: array<u32, 8>,
            strides: array<u32, 8>,
        }}

{inputs_decl}
        @group(0) @binding({binding_out}) var<storage, read_write> out: array<{wgsl_type}>;
        @group(0) @binding({binding_layouts}) var<storage, read> layout_infos: array<LayoutInfo>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let idx = global_id.x;
            if (idx >= arrayLength(&out)) {{
                return;
            }}
            
            let out_layout = layout_infos[{binding_out}];
            
            var temp = idx;
            var coords = array<u32, 8>();
            for (var d: u32 = 0u; d < out_layout.ndim; d = d + 1u) {{
                coords[d] = temp / out_layout.strides[d];
                temp = temp % out_layout.strides[d];
            }}

{offset_calcs}
            out[idx] = {expr_str};
        }}
        "#,
        inputs_decl = inputs_decl,
        wgsl_type = wgsl_type,
        binding_out = num_inputs,
        binding_layouts = num_inputs + 1,
        offset_calcs = offset_calcs,
        expr_str = expr_str
    );

    // 6. Create cache key
    let key = format!("fused_{}_{}", expr_str, wgsl_type);

    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");

    // 7. Bind Group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let mut entries = Vec::with_capacity(num_inputs + 2);
    for (i, input) in inputs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32,
            resource: input.storage().buffer.raw().as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: num_inputs as u32,
        resource: output.buffer.raw().as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: (num_inputs + 1) as u32,
        resource: layout_buf.as_entire_binding(),
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fused-bind-group"),
        layout: &bind_group_layout,
        entries: &entries,
    });

    // 8. Dispatch
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused-encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = out_layout.numel().div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}
