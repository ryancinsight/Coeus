#![allow(clippy::too_many_arguments)]

use super::PoolParams;
use crate::backend::WgpuScalar;
use crate::kernels::cache::PIPELINE_CACHE;
use crate::kernels::layout::GpuLayoutInfo;

pub fn dispatch_max_pool3d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &coeus_core::Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &wgpu::Buffer,
    output_layout: &coeus_core::Layout,
    out_numel: usize,
) {
    let ctx = crate::backend::get_wgpu_context();

    let in_layout_gpu = GpuLayoutInfo::from_layout(input_layout);
    let out_layout_gpu = GpuLayoutInfo::from_layout(output_layout);

    let params_data = PoolParams {
        kernel_size: kernel_size as u32,
        stride: stride as u32,
        padding: padding as u32,
        dilation: dilation as u32,
    };

    let in_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let out_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let params_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue
        .write_buffer(&in_layout_buf, 0, bytemuck::bytes_of(&in_layout_gpu));
    ctx.queue
        .write_buffer(&out_layout_buf, 0, bytemuck::bytes_of(&out_layout_gpu));
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params_data));

    let wgsl_type = T::WGSL_TYPE;
    let min_val = match wgsl_type {
        "i32" => "-2147483648",
        "u32" => "0u",
        _ => "-1e38",
    };
    let zero_val = match wgsl_type {
        "i32" => "0",
        "u32" => "0u",
        _ => "0.0",
    };

    let shader_src = format!(
        r#"
        struct LayoutInfo {{
            offset: u32,
            ndim: u32,
            shape: array<u32, 8>,
            strides: array<u32, 8>,
        }}

        struct PoolParams {{
            kernel_size: u32,
            stride: u32,
            padding: u32,
            dilation: u32,
        }}

        @group(0) @binding(0) var<storage, read> input: array<{wgsl_type}>;
        @group(0) @binding(1) var<storage, read_write> output: array<{wgsl_type}>;
        @group(0) @binding(2) var<storage, read> input_layout: LayoutInfo;
        @group(0) @binding(3) var<storage, read> output_layout: LayoutInfo;
        @group(0) @binding(4) var<storage, read> params: PoolParams;

        fn get_physical_index(ly: LayoutInfo, n: u32, c: u32, d: u32, h: u32, w: u32) -> u32 {{
            var idx = ly.offset;
            if (ly.ndim > 0u) {{ idx = idx + n * ly.strides[0]; }}
            if (ly.ndim > 1u) {{ idx = idx + c * ly.strides[1]; }}
            if (ly.ndim > 2u) {{ idx = idx + d * ly.strides[2]; }}
            if (ly.ndim > 3u) {{ idx = idx + h * ly.strides[3]; }}
            if (ly.ndim > 4u) {{ idx = idx + w * ly.strides[4]; }}
            return idx;
        }}

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let idx = global_id.x;
            let w_out = output_layout.shape[4];
            let h_out = output_layout.shape[3];
            let d_out = output_layout.shape[2];
            let c = output_layout.shape[1];
            let n = output_layout.shape[0];
            let out_numel = n * c * d_out * h_out * w_out;

            if (idx >= out_numel) {{
                return;
            }}

            let ow = idx % w_out;
            let temp1 = idx / w_out;
            let oh = temp1 % h_out;
            let temp2 = temp1 / h_out;
            let od = temp2 % d_out;
            let temp3 = temp2 / d_out;
            let ci = temp3 % c;
            let ni = temp3 / c;

            let d_in_limit = input_layout.shape[2];
            let h_in_limit = input_layout.shape[3];
            let w_in_limit = input_layout.shape[4];

            var max_val: {wgsl_type} = {min_val};
            var has_val = false;

            let kernel_size = params.kernel_size;
            let stride = params.stride;
            let padding = params.padding;
            let dilation = params.dilation;

            for (var ikd: u32 = 0u; ikd < kernel_size; ikd = ikd + 1u) {{
                let d_in = i32(od) * i32(stride) + i32(ikd) * i32(dilation) - i32(padding);
                if (d_in >= 0 && u32(d_in) < d_in_limit) {{
                    for (var ikh: u32 = 0u; ikh < kernel_size; ikh = ikh + 1u) {{
                        let h_in = i32(oh) * i32(stride) + i32(ikh) * i32(dilation) - i32(padding);
                        if (h_in >= 0 && u32(h_in) < h_in_limit) {{
                            for (var ikw: u32 = 0u; ikw < kernel_size; ikw = ikw + 1u) {{
                                let w_in = i32(ow) * i32(stride) + i32(ikw) * i32(dilation) - i32(padding);
                                if (w_in >= 0 && u32(w_in) < w_in_limit) {{
                                    let input_idx = get_physical_index(input_layout, ni, ci, u32(d_in), u32(h_in), u32(w_in));
                                    let val = input[input_idx];
                                    if (!has_val) {{
                                        max_val = val;
                                        has_val = true;
                                    }} else if (val > max_val) {{
                                        max_val = val;
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}

            let output_idx = get_physical_index(output_layout, ni, ci, od, oh, ow);
            if (has_val) {{
                output[output_idx] = max_val;
            }} else {{
                output[output_idx] = {zero_val};
            }}
        }}
    "#,
        wgsl_type = wgsl_type,
        min_val = min_val,
        zero_val = zero_val,
    );

    let key = format!("max_pool3d_{}", wgsl_type);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("max_pool3d-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: in_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("max_pool3d-encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("max_pool3d-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = out_numel.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

pub fn dispatch_max_pool3d_backward<T: WgpuScalar>(
    grad_out: &wgpu::Buffer,
    grad_out_layout: &coeus_core::Layout,
    input: &wgpu::Buffer,
    input_layout: &coeus_core::Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &wgpu::Buffer,
    grad_input_layout: &coeus_core::Layout,
    in_numel: usize,
) {
    let ctx = crate::backend::get_wgpu_context();

    let go_layout_gpu = GpuLayoutInfo::from_layout(grad_out_layout);
    let in_layout_gpu = GpuLayoutInfo::from_layout(input_layout);
    let gi_layout_gpu = GpuLayoutInfo::from_layout(grad_input_layout);

    let params_data = PoolParams {
        kernel_size: kernel_size as u32,
        stride: stride as u32,
        padding: padding as u32,
        dilation: dilation as u32,
    };

    let go_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let in_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let gi_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let params_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue
        .write_buffer(&go_layout_buf, 0, bytemuck::bytes_of(&go_layout_gpu));
    ctx.queue
        .write_buffer(&in_layout_buf, 0, bytemuck::bytes_of(&in_layout_gpu));
    ctx.queue
        .write_buffer(&gi_layout_buf, 0, bytemuck::bytes_of(&gi_layout_gpu));
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params_data));

    let wgsl_type = T::WGSL_TYPE;
    let min_val = match wgsl_type {
        "i32" => "-2147483648",
        "u32" => "0u",
        _ => "-1e38",
    };
    let zero_val = match wgsl_type {
        "i32" => "0",
        "u32" => "0u",
        _ => "0.0",
    };

    let shader_src = format!(
        r#"
        struct LayoutInfo {{
            offset: u32,
            ndim: u32,
            shape: array<u32, 8>,
            strides: array<u32, 8>,
        }}

        struct PoolParams {{
            kernel_size: u32,
            stride: u32,
            padding: u32,
            dilation: u32,
        }}

        @group(0) @binding(0) var<storage, read> grad_out: array<{wgsl_type}>;
        @group(0) @binding(1) var<storage, read> input: array<{wgsl_type}>;
        @group(0) @binding(2) var<storage, read_write> grad_input: array<{wgsl_type}>;
        @group(0) @binding(3) var<storage, read> go_layout: LayoutInfo;
        @group(0) @binding(4) var<storage, read> in_layout: LayoutInfo;
        @group(0) @binding(5) var<storage, read> gi_layout: LayoutInfo;
        @group(0) @binding(6) var<storage, read> params: PoolParams;

        fn get_physical_index(ly: LayoutInfo, n: u32, c: u32, d: u32, h: u32, w: u32) -> u32 {{
            var idx = ly.offset;
            if (ly.ndim > 0u) {{ idx = idx + n * ly.strides[0]; }}
            if (ly.ndim > 1u) {{ idx = idx + c * ly.strides[1]; }}
            if (ly.ndim > 2u) {{ idx = idx + d * ly.strides[2]; }}
            if (ly.ndim > 3u) {{ idx = idx + h * ly.strides[3]; }}
            if (ly.ndim > 4u) {{ idx = idx + w * ly.strides[4]; }}
            return idx;
        }}

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let idx = global_id.x;
            let w = gi_layout.shape[4];
            let h = gi_layout.shape[3];
            let d = gi_layout.shape[2];
            let c = gi_layout.shape[1];
            let n = gi_layout.shape[0];
            let in_numel = n * c * d * h * w;

            if (idx >= in_numel) {{
                return;
            }}

            let wi = idx % w;
            let temp1 = idx / w;
            let hi = temp1 % h;
            let temp2 = temp1 / h;
            let di = temp2 % d;
            let temp3 = temp2 / d;
            let ci = temp3 % c;
            let ni = temp3 / c;

            let my_idx = get_physical_index(in_layout, ni, ci, di, hi, wi);
            let my_val = input[my_idx];

            var sum: {wgsl_type} = {zero_val};

            let kernel_size = params.kernel_size;
            let stride = params.stride;
            let padding = params.padding;
            let dilation = params.dilation;

            let d_out_limit = go_layout.shape[2];
            let h_out_limit = go_layout.shape[3];
            let w_out_limit = go_layout.shape[4];

            for (var ikd: u32 = 0u; ikd < kernel_size; ikd = ikd + 1u) {{
                let numer_d = i32(di) + i32(padding) - i32(ikd) * i32(dilation);
                if (numer_d >= 0 && numer_d % i32(stride) == 0) {{
                    let od = u32(numer_d / i32(stride));
                    if (od < d_out_limit) {{
                        for (var ikh: u32 = 0u; ikh < kernel_size; ikh = ikh + 1u) {{
                            let numer_h = i32(hi) + i32(padding) - i32(ikh) * i32(dilation);
                            if (numer_h >= 0 && numer_h % i32(stride) == 0) {{
                                let oh = u32(numer_h / i32(stride));
                                if (oh < h_out_limit) {{
                                    for (var ikw: u32 = 0u; ikw < kernel_size; ikw = ikw + 1u) {{
                                        let numer_w = i32(wi) + i32(padding) - i32(ikw) * i32(dilation);
                                        if (numer_w >= 0 && numer_w % i32(stride) == 0) {{
                                            let ow = u32(numer_w / i32(stride));
                                            if (ow < w_out_limit) {{
                                                var max_val: {wgsl_type} = {min_val};
                                                var has_val = false;
                                                var max_d = 0u;
                                                var max_h = 0u;
                                                var max_w = 0u;

                                                for (var jkd: u32 = 0u; jkd < kernel_size; jkd = jkd + 1u) {{
                                                    let d_in = i32(od) * i32(stride) + i32(jkd) * i32(dilation) - i32(padding);
                                                    if (d_in >= 0 && u32(d_in) < d) {{
                                                        for (var jkh: u32 = 0u; jkh < kernel_size; jkh = jkh + 1u) {{
                                                            let h_in = i32(oh) * i32(stride) + i32(jkh) * i32(dilation) - i32(padding);
                                                            if (h_in >= 0 && u32(h_in) < h) {{
                                                                for (var jkw: u32 = 0u; jkw < kernel_size; jkw = jkw + 1u) {{
                                                                    let w_in = i32(ow) * i32(stride) + i32(jkw) * i32(dilation) - i32(padding);
                                                                    if (w_in >= 0 && u32(w_in) < w) {{
                                                                        let input_idx = get_physical_index(in_layout, ni, ci, u32(d_in), u32(h_in), u32(w_in));
                                                                        let val = input[input_idx];
                                                                        if (!has_val || val > max_val) {{
                                                                            max_val = val;
                                                                            max_d = u32(d_in);
                                                                            max_h = u32(h_in);
                                                                            max_w = u32(w_in);
                                                                            has_val = true;
                                                                        }}
                                                                    }}
                                                                }}
                                                            }}
                                                        }}
                                                    }}
                                                }}

                                                if (has_val && max_d == di && max_h == hi && max_w == wi && my_val == max_val) {{
                                                    let go_idx = get_physical_index(go_layout, ni, ci, od, oh, ow);
                                                    sum = sum + grad_out[go_idx];
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}

            let gi_idx = get_physical_index(gi_layout, ni, ci, di, hi, wi);
            grad_input[gi_idx] = grad_input[gi_idx] + sum;
        }}
        "#,
        wgsl_type = wgsl_type,
        min_val = min_val,
        zero_val = zero_val,
    );

    let key = format!("max_pool3d_backward_{}", wgsl_type);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("max_pool3d_backward-bind-group"),
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
                resource: grad_input.as_entire_binding(),
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
            label: Some("max_pool3d_backward-encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("max_pool3d_backward-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = in_numel.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}
