use super::cache::PIPELINE_CACHE;
use super::layout::GpuLayoutInfo;
use crate::backend::{WgpuBackend, WgpuBackendError, WgpuScalar};
use crate::storage::WgpuStorage;
use coeus_ops::fuse::ExprNode;
use std::collections::HashMap;

pub(crate) mod validation;

use validation::{validate_bindings, validate_reduction, validate_storage_bindings, OPERATION};

/// Dispatch a WGSL shader for fused element-wise and reduction along an axis.
///
/// # Errors
///
/// Returns [`WgpuBackendError`] when expression metadata, tensor layouts,
/// dispatch arithmetic, or adapter resource limits reject the operation.
pub fn dispatch_fused_reduce<T: WgpuScalar, E: ExprNode<T, WgpuBackend>>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
    c: &mut WgpuStorage<T>,
    c_layout: &coeus_core::Layout,
) -> Result<(), WgpuBackendError> {
    let mut inputs = Vec::new();
    expr.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(WgpuBackendError::Validation(
            coeus_core::BackendError::Storage {
                operation: OPERATION,
                reason: "expression contains no tensor inputs".to_string(),
            },
        ));
    }
    let expression_shape = expr.shape()?.ok_or_else(|| {
        WgpuBackendError::Validation(coeus_core::BackendError::Storage {
            operation: OPERATION,
            reason: "expression has no tensor input from which to derive its shape".to_string(),
        })
    })?;
    let num_inputs = inputs.len();

    let ctx = crate::backend::get_wgpu_context();
    let dispatch = validate_reduction(&expression_shape, op, axis, c_layout, &ctx.device.limits())?;
    let bindings = validate_bindings(num_inputs, &ctx.device.limits())?;
    validate_storage_bindings(
        inputs
            .iter()
            .map(|input| input.storage().buffer.raw().size()),
        c.buffer.raw().size(),
        &ctx.device.limits(),
    )?;
    let wgsl_type = T::WGSL_TYPE;

    // 2. Build input pointer to index map
    let mut input_map = HashMap::new();
    for (i, &input) in inputs.iter().enumerate() {
        input_map.insert(std::ptr::from_ref(input), i);
    }
    // 3. Generate the shader expression string
    let expr_str = expr.to_shader_expr(&input_map);

    // 4. Create the unified layout buffer
    let mut layouts_gpu = Vec::with_capacity(num_inputs + 1);
    for input in &inputs {
        layouts_gpu.push(
            GpuLayoutInfo::try_from_layout(input.layout())
                .map_err(|error| WgpuBackendError::Layout(error.into()))?,
        );
    }
    // We add c_layout as the last one to decode output coordinates
    layouts_gpu.push(
        GpuLayoutInfo::try_from_layout(c_layout)
            .map_err(|error| WgpuBackendError::Layout(error.into()))?,
    );

    let layout_buf = crate::backend::PooledMetadataBuffer::new();
    ctx.queue
        .write_buffer(&layout_buf, 0, bytemuck::cast_slice(&layouts_gpu));

    let axis_info = [dispatch.axis, dispatch.axis_length];
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
            i,
            i,
            dispatch.expression_rank,
            dispatch.expression_rank,
            i,
            i,
            dispatch.expression_rank,
            i,
            i,
            i,
            i,
            i,
            i,
            i
        ));
    }

    let (init_expr, update_expr) = match op {
        coeus_ops::ReductionOp::Sum => (T::WGSL_ZERO, "acc = acc + val;"),
        coeus_ops::ReductionOp::Prod => (T::WGSL_ONE, "acc = acc * val;"),
        coeus_ops::ReductionOp::Mean => (T::WGSL_ZERO, "acc = acc + val;"),
        coeus_ops::ReductionOp::Max => (T::WGSL_LOWEST, "acc = max(acc, val);"),
        coeus_ops::ReductionOp::Min => (T::WGSL_HIGHEST, "acc = min(acc, val);"),
    };
    let final_expr = match op {
        coeus_ops::ReductionOp::Mean => format!("acc / {wgsl_type}(axis_len)"),
        _ => "acc".to_string(),
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
            
            out[idx] = {final_expr};
        }}
        "#,
        inputs_decl = inputs_decl,
        wgsl_type = wgsl_type,
        binding_out = bindings.output,
        binding_layouts = bindings.layouts,
        binding_axis = bindings.axis,
        binding_c = bindings.output,
        init_expr = init_expr,
        offset_calcs = offset_calcs,
        expr_str = expr_str,
        update_expr = update_expr,
        final_expr = final_expr
    );

    // 6. Create cache key
    let key = format!("fused_reduce_{:?}_{}_{}", op, expr_str, wgsl_type);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
    // 7. Bind Group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let mut entries = Vec::with_capacity(num_inputs + 3);
    for (i, input) in inputs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: u32::try_from(i).expect("invariant: validated input binding fits u32"),
            resource: input.storage().buffer.raw().as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: bindings.output,
        resource: c.buffer.raw().as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: bindings.layouts,
        resource: layout_buf.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: bindings.axis,
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

        compute_pass.dispatch_workgroups(dispatch.workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    Ok(())
}
