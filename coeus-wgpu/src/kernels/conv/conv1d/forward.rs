use crate::backend::WgpuScalar;
use crate::kernels::cache::PIPELINE_CACHE;
use crate::kernels::layout::GpuLayoutInfo;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ConvParams {
    pub(crate) stride: u32,
    pub(crate) padding: u32,
    pub(crate) dilation: u32,
    pub(crate) has_bias: u32,
}

pub fn dispatch_conv1d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    output: &wgpu::Buffer,
    input_layout: &coeus_core::Layout,
    weight_layout: &coeus_core::Layout,
    output_layout: &coeus_core::Layout,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_numel: usize,
) {
    let ctx = crate::backend::get_wgpu_context();

    let in_layout_gpu = GpuLayoutInfo::from_layout(input_layout);
    let w_layout_gpu = GpuLayoutInfo::from_layout(weight_layout);
    let out_layout_gpu = GpuLayoutInfo::from_layout(output_layout);

    let has_bias = if bias.is_some() { 1u32 } else { 0u32 };
    let params_data = ConvParams {
        stride: stride as u32,
        padding: padding as u32,
        dilation: dilation as u32,
        has_bias,
    };

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
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params_data));

    let dummy_bias = crate::backend::PooledMetadataBuffer::new();
    let bias_buf = bias.unwrap_or(&dummy_bias);

    let shader_src = format!(
        r#"
        struct LayoutInfo {{ offset: u32, ndim: u32, shape: array<u32, 8>, strides: array<u32, 8>, }}

        struct ConvParams {{ stride: u32, padding: u32, dilation: u32, has_bias: u32, }}

        @group(0) @binding(0) var<storage, read> input: array<{wgsl_type}>;
        @group(0) @binding(1) var<storage, read> weight: array<{wgsl_type}>;
        @group(0) @binding(2) var<storage, read> bias: array<{wgsl_type}>;
        @group(0) @binding(3) var<storage, read_write> output: array<{wgsl_type}>;
        @group(0) @binding(4) var<storage, read> input_layout: LayoutInfo;
        @group(0) @binding(5) var<storage, read> weight_layout: LayoutInfo;
        @group(0) @binding(6) var<storage, read> output_layout: LayoutInfo;
        @group(0) @binding(7) var<storage, read> params: ConvParams;

        fn get_physical_index_3d(ly: LayoutInfo, c0: u32, c1: u32, c2: u32) -> u32 {{
            return ly.offset + c0 * ly.strides[0] + c1 * ly.strides[1] + c2 * ly.strides[2];
        }}

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let i = global_id.x;
            let l_out = output_layout.shape[2];
            let c_out = weight_layout.shape[0];
            let n = input_layout.shape[0];
            let c_in = input_layout.shape[1];
            let l = input_layout.shape[2];
            let k = weight_layout.shape[2];

            let out_numel = n * c_out * l_out;
            if (i >= out_numel) {{
                return;
            }}

            let ol = i % l_out;
            let temp = i / l_out;
            let oc = temp % c_out;
            let ni = temp / c_out;

            var sum = {wgsl_zero};
            for (var ic: u32 = 0u; ic < c_in; ic = ic + 1u) {{
                for (var ik: u32 = 0u; ik < k; ik = ik + 1u) {{
                    let input_idx = i32(ol) * i32(params.stride) + i32(ik) * i32(params.dilation) - i32(params.padding);
                    if (input_idx >= 0 && u32(input_idx) < l) {{
                        let in_idx = get_physical_index_3d(input_layout, ni, ic, u32(input_idx));
                        let w_idx = get_physical_index_3d(weight_layout, oc, ic, ik);
                        sum = sum + input[in_idx] * weight[w_idx];
                    }}
                }}
            }}

            if (params.has_bias != 0u) {{
                sum = sum + bias[oc];
            }}

            let out_idx = get_physical_index_3d(output_layout, ni, oc, ol);
            output[out_idx] = sum;
        }}
        "#,
        wgsl_type = T::WGSL_TYPE,
        wgsl_zero = T::WGSL_ZERO,
    );

    let key = format!("conv1d_{}", T::WGSL_TYPE);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("conv1d-bind-group"),
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
                resource: bias_buf.as_entire_binding(),
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
            label: Some("conv1d-encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv1d-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = out_numel.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}
