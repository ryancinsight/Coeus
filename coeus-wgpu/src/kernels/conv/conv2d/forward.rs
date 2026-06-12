use crate::backend::WgpuScalar;
use crate::kernels::cache::PIPELINE_CACHE;
use crate::kernels::layout::GpuLayoutInfo;

pub struct Conv2dDispatch<'a> {
    pub input: &'a wgpu::Buffer,
    pub weight: &'a wgpu::Buffer,
    pub bias: Option<&'a wgpu::Buffer>,
    pub output: &'a wgpu::Buffer,
    pub input_layout: &'a coeus_core::Layout,
    pub weight_layout: &'a coeus_core::Layout,
    pub output_layout: &'a coeus_core::Layout,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out_numel: usize,
}

pub fn dispatch_conv2d<T: WgpuScalar>(request: Conv2dDispatch<'_>) {
    let Conv2dDispatch {
        input,
        weight,
        bias,
        output,
        input_layout,
        weight_layout,
        output_layout,
        stride,
        padding,
        dilation,
        out_numel,
    } = request;
    let ctx = crate::backend::get_wgpu_context();

    let in_layout_gpu = GpuLayoutInfo::from_layout(input_layout);
    let w_layout_gpu = GpuLayoutInfo::from_layout(weight_layout);
    let out_layout_gpu = GpuLayoutInfo::from_layout(output_layout);
    let has_bias = if bias.is_some() { 1u32 } else { 0u32 };
    let params_data = [stride as u32, padding as u32, dilation as u32, has_bias];

    let in_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let w_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let out_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let params_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue
        .write_buffer(&in_layout_buf, 0, bytemuck::bytes_of(&in_layout_gpu));
    ctx.queue
        .write_buffer(&w_layout_buf, 0, bytemuck::bytes_of(&w_layout_gpu));
    ctx.queue
        .write_buffer(&out_layout_buf, 0, bytemuck::bytes_of(&out_layout_gpu));
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

    let dummy_bias_buf = crate::backend::PooledMetadataBuffer::new();
    let bias_buf_ref = bias.unwrap_or(&dummy_bias_buf);

    let shader_src = format!(
        r#"
        struct LayoutInfo {{ offset: u32, ndim: u32, shape: array<u32, 8>, strides: array<u32, 8>, }}

        @group(0) @binding(0) var<storage, read> input: array<{wgsl_type}>;
        @group(0) @binding(1) var<storage, read> weight: array<{wgsl_type}>;
        @group(0) @binding(2) var<storage, read> bias: array<{wgsl_type}>;
        @group(0) @binding(3) var<storage, read_write> output: array<{wgsl_type}>;
        @group(0) @binding(4) var<storage, read> in_layout: LayoutInfo;
        @group(0) @binding(5) var<storage, read> w_layout: LayoutInfo;
        @group(0) @binding(6) var<storage, read> out_layout: LayoutInfo;
        @group(0) @binding(7) var<storage, read> params: array<u32, 4>; // stride, padding, dilation, has_bias

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let idx = global_id.x;
            let out_numel = arrayLength(&output);
            if (idx >= out_numel) {{
                return;
            }}

            let stride_val = params[0];
            let padding_val = params[1];
            let dilation_val = params[2];
            let has_bias = params[3];

            let h_out = out_layout.shape[2];
            let w_out = out_layout.shape[3];
            let c_out = w_layout.shape[0];
            let c_in = in_layout.shape[1];
            let h = in_layout.shape[2];
            let w = in_layout.shape[3];
            let kh = w_layout.shape[2];
            let kw = w_layout.shape[3];

            let ow = idx % w_out;
            let temp1 = idx / w_out;
            let oh = temp1 % h_out;
            let temp2 = temp1 / h_out;
            let oc = temp2 % c_out;
            let ni = temp2 / c_out;

            var sum = {wgsl_zero};
            for (var ic: u32 = 0u; ic < c_in; ic = ic + 1u) {{
                for (var ikh: u32 = 0u; ikh < kh; ikh = ikh + 1u) {{
                    let h_in = i32(oh) * i32(stride_val) + i32(ikh) * i32(dilation_val) - i32(padding_val);
                    if (h_in >= 0 && u32(h_in) < h) {{
                        for (var ikw: u32 = 0u; ikw < kw; ikw = ikw + 1u) {{
                            let w_in = i32(ow) * i32(stride_val) + i32(ikw) * i32(dilation_val) - i32(padding_val);
                            if (w_in >= 0 && u32(w_in) < w) {{
                                let in_idx = in_layout.offset + ni * in_layout.strides[0] + ic * in_layout.strides[1] + u32(h_in) * in_layout.strides[2] + u32(w_in) * in_layout.strides[3];
                                let w_idx = w_layout.offset + oc * w_layout.strides[0] + ic * w_layout.strides[1] + ikh * w_layout.strides[2] + ikw * w_layout.strides[3];
                                sum = sum + input[in_idx] * weight[w_idx];
                            }}
                        }}
                    }}
                }}
            }}

            if (has_bias != 0u) {{
                sum = sum + bias[oc];
            }}

            let out_idx = out_layout.offset + ni * out_layout.strides[0] + oc * out_layout.strides[1] + oh * out_layout.strides[2] + ow * out_layout.strides[3];
            output[out_idx] = sum;
        }}
        "#,
        wgsl_type = T::WGSL_TYPE,
        wgsl_zero = T::WGSL_ZERO,
    );

    let key = format!("conv2d_{}", T::WGSL_TYPE);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("conv2d-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bias_buf_ref.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: in_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: w_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: out_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv2d-encoder"),
        });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv2d-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = out_numel.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
}
