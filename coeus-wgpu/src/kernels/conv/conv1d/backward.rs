use super::forward::ConvParams;
use crate::backend::WgpuScalar;
use crate::kernels::cache::PIPELINE_CACHE;
use crate::kernels::layout::GpuLayoutInfo;

pub struct Conv1dBackwardDispatch<'a> {
    pub grad_out: &'a wgpu::Buffer,
    pub grad_out_layout: &'a coeus_core::Layout,
    pub input: &'a wgpu::Buffer,
    pub input_layout: &'a coeus_core::Layout,
    pub weight: &'a wgpu::Buffer,
    pub weight_layout: &'a coeus_core::Layout,
    pub grad_input: Option<&'a wgpu::Buffer>,
    pub grad_input_layout: &'a coeus_core::Layout,
    pub grad_weight: Option<&'a wgpu::Buffer>,
    pub grad_weight_layout: &'a coeus_core::Layout,
    pub grad_bias: Option<&'a wgpu::Buffer>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

pub fn dispatch_conv1d_backward<T: WgpuScalar>(request: Conv1dBackwardDispatch<'_>) {
    let Conv1dBackwardDispatch {
        grad_out,
        grad_out_layout,
        input,
        input_layout,
        weight,
        weight_layout,
        grad_input,
        grad_input_layout,
        grad_weight,
        grad_weight_layout,
        grad_bias,
        stride,
        padding,
        dilation,
    } = request;
    let ctx = crate::backend::get_wgpu_context();

    let go_layout_gpu = GpuLayoutInfo::from_layout(grad_out_layout);
    let in_layout_gpu = GpuLayoutInfo::from_layout(input_layout);
    let w_layout_gpu = GpuLayoutInfo::from_layout(weight_layout);
    let gi_layout_gpu = GpuLayoutInfo::from_layout(grad_input_layout);
    let gw_layout_gpu = GpuLayoutInfo::from_layout(grad_weight_layout);

    let params_data = ConvParams {
        stride: stride as u32,
        padding: padding as u32,
        dilation: dilation as u32,
        has_bias: 0,
    };

    let go_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let in_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let w_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let params_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue
        .write_buffer(&go_layout_buf, 0, bytemuck::bytes_of(&go_layout_gpu));
    ctx.queue
        .write_buffer(&in_layout_buf, 0, bytemuck::bytes_of(&in_layout_gpu));
    ctx.queue
        .write_buffer(&w_layout_buf, 0, bytemuck::bytes_of(&w_layout_gpu));
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params_data));

    // 1. Grad Input
    if let Some(gi) = grad_input {
        let gi_layout_buf = crate::backend::PooledMetadataBuffer::new();
        ctx.queue
            .write_buffer(&gi_layout_buf, 0, bytemuck::bytes_of(&gi_layout_gpu));

        let shader_src = format!(
            r#"
            struct LayoutInfo {{ offset: u32, ndim: u32, shape: array<u32, 8>, strides: array<u32, 8>, }}

            struct ConvParams {{ stride: u32, padding: u32, dilation: u32, has_bias: u32, }}

            @group(0) @binding(0) var<storage, read> grad_out: array<{wgsl_type}>;
            @group(0) @binding(1) var<storage, read> weight: array<{wgsl_type}>;
            @group(0) @binding(2) var<storage, read_write> grad_input: array<{wgsl_type}>;
            @group(0) @binding(3) var<storage, read> go_layout: LayoutInfo;
            @group(0) @binding(4) var<storage, read> w_layout: LayoutInfo;
            @group(0) @binding(5) var<storage, read> gi_layout: LayoutInfo;
            @group(0) @binding(6) var<storage, read> params: ConvParams;

            fn get_physical_index_3d(ly: LayoutInfo, c0: u32, c1: u32, c2: u32) -> u32 {{
                return ly.offset + c0 * ly.strides[0] + c1 * ly.strides[1] + c2 * ly.strides[2];
            }}

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let i = global_id.x;
                let n = gi_layout.shape[0];
                let c_in = gi_layout.shape[1];
                let l = gi_layout.shape[2];
                let c_out = w_layout.shape[0];
                let k = w_layout.shape[2];
                let l_out = go_layout.shape[2];

                let numel_in = n * c_in * l;
                if (i >= numel_in) {{
                    return;
                }}

                let li = i % l;
                let temp = i / l;
                let ic = temp % c_in;
                let ni = temp / c_in;

                var sum = {wgsl_zero};
                for (var oc: u32 = 0u; oc < c_out; oc = oc + 1u) {{
                    for (var ik: u32 = 0u; ik < k; ik = ik + 1u) {{
                        let numer = i32(li) + i32(params.padding) - i32(ik) * i32(params.dilation);
                        if (numer >= 0 && numer % i32(params.stride) == 0) {{
                            let ol = u32(numer / i32(params.stride));
                            if (ol < l_out) {{
                                let go_idx = get_physical_index_3d(go_layout, ni, oc, ol);
                                let w_idx = get_physical_index_3d(w_layout, oc, ic, ik);
                                sum = sum + grad_out[go_idx] * weight[w_idx];
                            }}
                        }}
                    }}
                }}

                let gi_idx = get_physical_index_3d(gi_layout, ni, ic, li);
                grad_input[gi_idx] = grad_input[gi_idx] + sum;
            }}
            "#,
            wgsl_type = T::WGSL_TYPE,
            wgsl_zero = T::WGSL_ZERO,
        );

        let key = format!("conv1d_back_gi_{}", T::WGSL_TYPE);
        let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv1d-back-gi-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gi.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: go_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: w_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gi_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("conv1d-back-gi-encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv1d-back-gi-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let numel_in = gi_layout_gpu.shape[0] * gi_layout_gpu.shape[1] * gi_layout_gpu.shape[2];
            let workgroups = (numel_in as usize).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    // 2. Grad Weight
    if let Some(gw) = grad_weight {
        let gw_layout_buf = crate::backend::PooledMetadataBuffer::new();
        ctx.queue
            .write_buffer(&gw_layout_buf, 0, bytemuck::bytes_of(&gw_layout_gpu));

        let shader_src = format!(
            r#"
            struct LayoutInfo {{ offset: u32, ndim: u32, shape: array<u32, 8>, strides: array<u32, 8>, }}

            struct ConvParams {{ stride: u32, padding: u32, dilation: u32, has_bias: u32, }}

            @group(0) @binding(0) var<storage, read> grad_out: array<{wgsl_type}>;
            @group(0) @binding(1) var<storage, read> input: array<{wgsl_type}>;
            @group(0) @binding(2) var<storage, read_write> grad_weight: array<{wgsl_type}>;
            @group(0) @binding(3) var<storage, read> go_layout: LayoutInfo;
            @group(0) @binding(4) var<storage, read> in_layout: LayoutInfo;
            @group(0) @binding(5) var<storage, read> gw_layout: LayoutInfo;
            @group(0) @binding(6) var<storage, read> params: ConvParams;

            fn get_physical_index_3d(ly: LayoutInfo, c0: u32, c1: u32, c2: u32) -> u32 {{
                return ly.offset + c0 * ly.strides[0] + c1 * ly.strides[1] + c2 * ly.strides[2];
            }}

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let i = global_id.x;
                let c_out = gw_layout.shape[0];
                let c_in = gw_layout.shape[1];
                let k = gw_layout.shape[2];
                let n = in_layout.shape[0];
                let l = in_layout.shape[2];
                let l_out = go_layout.shape[2];

                let numel_w = c_out * c_in * k;
                if (i >= numel_w) {{
                    return;
                }}

                let ik = i % k;
                let temp = i / k;
                let ic = temp % c_in;
                let oc = temp / c_in;

                var sum = {wgsl_zero};
                for (var ni: u32 = 0u; ni < n; ni = ni + 1u) {{
                    for (var ol: u32 = 0u; ol < l_out; ol = ol + 1u) {{
                        let l_in = i32(ol) * i32(params.stride) + i32(ik) * i32(params.dilation) - i32(params.padding);
                        if (l_in >= 0 && u32(l_in) < l) {{
                            let go_idx = get_physical_index_3d(go_layout, ni, oc, ol);
                            let input_idx = get_physical_index_3d(in_layout, ni, ic, u32(l_in));
                            sum = sum + grad_out[go_idx] * input[input_idx];
                        }}
                    }}
                }}

                let gw_idx = get_physical_index_3d(gw_layout, oc, ic, ik);
                grad_weight[gw_idx] = grad_weight[gw_idx] + sum;
            }}
            "#,
            wgsl_type = T::WGSL_TYPE,
            wgsl_zero = T::WGSL_ZERO,
        );

        let key = format!("conv1d_back_gw_{}", T::WGSL_TYPE);
        let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv1d-back-gw-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: go_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: in_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gw_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("conv1d-back-gw-encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv1d-back-gw-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let numel_w = gw_layout_gpu.shape[0] * gw_layout_gpu.shape[1] * gw_layout_gpu.shape[2];
            let workgroups = (numel_w as usize).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    // 3. Grad Bias
    if let Some(gb) = grad_bias {
        let shader_src = format!(
            r#"
            struct LayoutInfo {{ offset: u32, ndim: u32, shape: array<u32, 8>, strides: array<u32, 8>, }}

            @group(0) @binding(0) var<storage, read> grad_out: array<{wgsl_type}>;
            @group(0) @binding(1) var<storage, read_write> grad_bias: array<{wgsl_type}>;
            @group(0) @binding(2) var<storage, read> go_layout: LayoutInfo;

            fn get_physical_index_3d(ly: LayoutInfo, c0: u32, c1: u32, c2: u32) -> u32 {{
                return ly.offset + c0 * ly.strides[0] + c1 * ly.strides[1] + c2 * ly.strides[2];
            }}

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let oc = global_id.x;
                let c_out = go_layout.shape[1];
                let n = go_layout.shape[0];
                let l_out = go_layout.shape[2];

                if (oc >= c_out) {{
                    return;
                }}

                var sum = {wgsl_zero};
                for (var ni: u32 = 0u; ni < n; ni = ni + 1u) {{
                    for (var ol: u32 = 0u; ol < l_out; ol = ol + 1u) {{
                        let go_idx = get_physical_index_3d(go_layout, ni, oc, ol);
                        sum = sum + grad_out[go_idx];
                    }}
                }}

                grad_bias[oc] = grad_bias[oc] + sum;
            }}
            "#,
            wgsl_type = T::WGSL_TYPE,
            wgsl_zero = T::WGSL_ZERO,
        );

        let key = format!("conv1d_back_gb_{}", T::WGSL_TYPE);
        let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv1d-back-gb-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: go_layout_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("conv1d-back-gb-encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv1d-back-gb-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (go_layout_gpu.shape[1] as usize).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }
}
