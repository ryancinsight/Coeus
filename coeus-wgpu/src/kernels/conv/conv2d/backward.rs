use crate::backend::WgpuScalar;
use crate::kernels::cache::PIPELINE_CACHE;
use crate::kernels::layout::GpuLayoutInfo;

pub fn dispatch_conv2d_backward<T: WgpuScalar>(
    grad_out: &wgpu::Buffer,
    grad_out_layout: &coeus_core::Layout,
    input: &wgpu::Buffer,
    input_layout: &coeus_core::Layout,
    weight: &wgpu::Buffer,
    weight_layout: &coeus_core::Layout,
    grad_input: Option<&wgpu::Buffer>,
    grad_input_layout: &coeus_core::Layout,
    grad_weight: Option<&wgpu::Buffer>,
    grad_weight_layout: &coeus_core::Layout,
    grad_bias: Option<&wgpu::Buffer>,
    stride: usize,
    padding: usize,
    dilation: usize,
) {
    let ctx = crate::backend::get_wgpu_context();

    let go_layout_gpu = GpuLayoutInfo::from_layout(grad_out_layout);
    let in_layout_gpu = GpuLayoutInfo::from_layout(input_layout);
    let w_layout_gpu = GpuLayoutInfo::from_layout(weight_layout);
    let gi_layout_gpu = GpuLayoutInfo::from_layout(grad_input_layout);
    let gw_layout_gpu = GpuLayoutInfo::from_layout(grad_weight_layout);

    let params_data = [stride as u32, padding as u32, dilation as u32, 0u32];

    let go_layout_buf = crate::backend::PooledMetadataBuffer::new();
    ctx.queue
        .write_buffer(&go_layout_buf, 0, bytemuck::bytes_of(&go_layout_gpu));

    let params_buf = crate::backend::PooledMetadataBuffer::new();
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

    if let Some(gi) = grad_input {
        let w_layout_buf = crate::backend::PooledMetadataBuffer::new();
        ctx.queue
            .write_buffer(&w_layout_buf, 0, bytemuck::bytes_of(&w_layout_gpu));

        let gi_layout_buf = crate::backend::PooledMetadataBuffer::new();
        ctx.queue
            .write_buffer(&gi_layout_buf, 0, bytemuck::bytes_of(&gi_layout_gpu));

        let shader_src = format!(
            r#"
            struct LayoutInfo {{ offset: u32, ndim: u32, shape: array<u32, 8>, strides: array<u32, 8>, }}

            @group(0) @binding(0) var<storage, read> grad_out: array<{wgsl_type}>;
            @group(0) @binding(1) var<storage, read> weight: array<{wgsl_type}>;
            @group(0) @binding(2) var<storage, read_write> grad_input: array<{wgsl_type}>;
            @group(0) @binding(3) var<storage, read> go_layout: LayoutInfo;
            @group(0) @binding(4) var<storage, read> w_layout: LayoutInfo;
            @group(0) @binding(5) var<storage, read> gi_layout: LayoutInfo;
            @group(0) @binding(6) var<storage, read> params: array<u32, 4>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                let n = gi_layout.shape[0];
                let c_in = gi_layout.shape[1];
                let h = gi_layout.shape[2];
                let w = gi_layout.shape[3];
                let c_out = w_layout.shape[0];
                let kh = w_layout.shape[2];
                let kw = w_layout.shape[3];
                let h_out = go_layout.shape[2];
                let w_out = go_layout.shape[3];

                let numel_in = n * c_in * h * w;
                if (idx >= numel_in) {{
                    return;
                }}

                let stride_val = params[0];
                let padding_val = params[1];
                let dilation_val = params[2];

                let wi = idx % w;
                let temp1 = idx / w;
                let hi = temp1 % h;
                let temp2 = temp1 / h;
                let ic = temp2 % c_in;
                let ni = temp2 / c_in;

                var sum = {wgsl_zero};
                for (var oc: u32 = 0u; oc < c_out; oc = oc + 1u) {{
                    for (var ikh: u32 = 0u; ikh < kh; ikh = ikh + 1u) {{
                        let numer_h = i32(hi) + i32(padding_val) - i32(ikh) * i32(dilation_val);
                        if (numer_h >= 0 && numer_h % i32(stride_val) == 0) {{
                            let oh = u32(numer_h / i32(stride_val));
                            if (oh < h_out) {{
                                for (var ikw: u32 = 0u; ikw < kw; ikw = ikw + 1u) {{
                                    let numer_w = i32(wi) + i32(padding_val) - i32(ikw) * i32(dilation_val);
                                    if (numer_w >= 0 && numer_w % i32(stride_val) == 0) {{
                                        let ow = u32(numer_w / i32(stride_val));
                                        if (ow < w_out) {{
                                            let go_idx = go_layout.offset + ni * go_layout.strides[0] + oc * go_layout.strides[1] + oh * go_layout.strides[2] + ow * go_layout.strides[3];
                                            let w_idx = w_layout.offset + oc * w_layout.strides[0] + ic * w_layout.strides[1] + ikh * w_layout.strides[2] + ikw * w_layout.strides[3];
                                            sum = sum + grad_out[go_idx] * weight[w_idx];
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}

                let gi_idx = gi_layout.offset + ni * gi_layout.strides[0] + ic * gi_layout.strides[1] + hi * gi_layout.strides[2] + wi * gi_layout.strides[3];
                grad_input[gi_idx] = grad_input[gi_idx] + sum;
            }}
            "#,
            wgsl_type = T::WGSL_TYPE,
            wgsl_zero = T::WGSL_ZERO,
        );

        let key = format!("conv2d_gi_{}", T::WGSL_TYPE);
        let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv2d-gi-bind-group"),
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
                label: Some("conv2d-gi-encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv2d-gi-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let numel_in = grad_input_layout.shape().iter().product::<usize>();
            let workgroups = numel_in.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    if let Some(gw) = grad_weight {
        let in_layout_buf = crate::backend::PooledMetadataBuffer::new();
        ctx.queue
            .write_buffer(&in_layout_buf, 0, bytemuck::bytes_of(&in_layout_gpu));

        let gw_layout_buf = crate::backend::PooledMetadataBuffer::new();
        ctx.queue
            .write_buffer(&gw_layout_buf, 0, bytemuck::bytes_of(&gw_layout_gpu));

        let shader_src = format!(
            r#"
            struct LayoutInfo {{ offset: u32, ndim: u32, shape: array<u32, 8>, strides: array<u32, 8>, }}

            @group(0) @binding(0) var<storage, read> grad_out: array<{wgsl_type}>;
            @group(0) @binding(1) var<storage, read> input: array<{wgsl_type}>;
            @group(0) @binding(2) var<storage, read_write> grad_weight: array<{wgsl_type}>;
            @group(0) @binding(3) var<storage, read> go_layout: LayoutInfo;
            @group(0) @binding(4) var<storage, read> in_layout: LayoutInfo;
            @group(0) @binding(5) var<storage, read> gw_layout: LayoutInfo;
            @group(0) @binding(6) var<storage, read> params: array<u32, 4>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                let c_out = gw_layout.shape[0];
                let c_in = gw_layout.shape[1];
                let kh = gw_layout.shape[2];
                let kw = gw_layout.shape[3];
                let n = in_layout.shape[0];
                let h = in_layout.shape[2];
                let w = in_layout.shape[3];
                let h_out = go_layout.shape[2];
                let w_out = go_layout.shape[3];

                let numel_w = c_out * c_in * kh * kw;
                if (idx >= numel_w) {{
                    return;
                }}

                let stride_val = params[0];
                let padding_val = params[1];
                let dilation_val = params[2];

                let ikw = idx % kw;
                let temp1 = idx / kw;
                let ikh = temp1 % kh;
                let temp2 = temp1 / kh;
                let ic = temp2 % c_in;
                let oc = temp2 / c_in;

                var sum = {wgsl_zero};
                for (var ni: u32 = 0u; ni < n; ni = ni + 1u) {{
                    for (var oh: u32 = 0u; oh < h_out; oh = oh + 1u) {{
                        let h_in = i32(oh) * i32(stride_val) + i32(ikh) * i32(dilation_val) - i32(padding_val);
                        if (h_in >= 0 && u32(h_in) < h) {{
                            for (var ow: u32 = 0u; ow < w_out; ow = ow + 1u) {{
                                let w_in = i32(ow) * i32(stride_val) + i32(ikw) * i32(dilation_val) - i32(padding_val);
                                if (w_in >= 0 && u32(w_in) < w) {{
                                    let go_idx = go_layout.offset + ni * go_layout.strides[0] + oc * go_layout.strides[1] + oh * go_layout.strides[2] + ow * go_layout.strides[3];
                                    let input_idx = in_layout.offset + ni * in_layout.strides[0] + ic * in_layout.strides[1] + u32(h_in) * in_layout.strides[2] + u32(w_in) * in_layout.strides[3];
                                    sum = sum + grad_out[go_idx] * input[input_idx];
                                }}
                            }}
                        }}
                    }}
                }}

                let gw_idx = gw_layout.offset + oc * gw_layout.strides[0] + ic * gw_layout.strides[1] + ikh * gw_layout.strides[2] + ikw * gw_layout.strides[3];
                grad_weight[gw_idx] = grad_weight[gw_idx] + sum;
            }}
            "#,
            wgsl_type = T::WGSL_TYPE,
            wgsl_zero = T::WGSL_ZERO,
        );

        let key = format!("conv2d_gw_{}", T::WGSL_TYPE);
        let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv2d-gw-bind-group"),
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
                label: Some("conv2d-gw-encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv2d-gw-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let numel_w = grad_weight_layout.shape().iter().product::<usize>();
            let workgroups = numel_w.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    if let Some(gb) = grad_bias {
        let shader_src = format!(
            r#"
            struct LayoutInfo {{ offset: u32, ndim: u32, shape: array<u32, 8>, strides: array<u32, 8>, }}

            @group(0) @binding(0) var<storage, read> grad_out: array<{wgsl_type}>;
            @group(0) @binding(1) var<storage, read_write> grad_bias: array<{wgsl_type}>;
            @group(0) @binding(2) var<storage, read> go_layout: LayoutInfo;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let oc = global_id.x;
                let c_out = go_layout.shape[1];
                if (oc >= c_out) {{
                    return;
                }}

                let n = go_layout.shape[0];
                let h_out = go_layout.shape[2];
                let w_out = go_layout.shape[3];

                var sum = {wgsl_zero};
                for (var ni: u32 = 0u; ni < n; ni = ni + 1u) {{
                    for (var oh: u32 = 0u; oh < h_out; oh = oh + 1u) {{
                        for (var ow: u32 = 0u; ow < w_out; ow = ow + 1u) {{
                            let go_idx = go_layout.offset + ni * go_layout.strides[0] + oc * go_layout.strides[1] + oh * go_layout.strides[2] + ow * go_layout.strides[3];
                            sum = sum + grad_out[go_idx];
                        }}
                    }}
                }}

                grad_bias[oc] = grad_bias[oc] + sum;
            }}
            "#,
            wgsl_type = T::WGSL_TYPE,
            wgsl_zero = T::WGSL_ZERO,
        );

        let key = format!("conv2d_gb_{}", T::WGSL_TYPE);
        let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv2d-gb-bind-group"),
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
                label: Some("conv2d-gb-encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv2d-gb-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let c_out = go_layout_gpu.shape[1];
            let workgroups = (c_out as usize).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }
}
