use super::cache::PIPELINE_CACHE;
use super::layout::GpuLayoutInfo;
use crate::backend::WgpuScalar;
use coeus_core::Layout;

const WORKGROUP_SIZE: usize = 256;

#[derive(Clone, Copy)]
enum KernelKind {
    Unfold1d,
    Fold1d,
    Unfold2d,
    Fold2d,
}

fn parameter(value: usize, name: &str) -> u32 {
    u32::try_from(value)
        .unwrap_or_else(|_| panic!("{name} exceeds the WGSL u32 index range: {value}"))
}

fn output_width(
    layout: &Layout,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let width = layout
        .shape()
        .last()
        .copied()
        .expect("unfold/fold requires a non-empty layout shape");
    let effective_kernel = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|value| value.checked_add(1))
        .expect("unfold/fold effective kernel exceeds usize");
    let padded = width
        .checked_add(
            padding
                .checked_mul(2)
                .expect("unfold/fold padding exceeds usize"),
        )
        .expect("unfold/fold padded width exceeds usize");
    padded
        .checked_sub(effective_kernel)
        .expect("unfold/fold kernel exceeds padded input")
        .checked_div(stride)
        .expect("unfold/fold stride must be non-zero")
        .checked_add(1)
        .expect("unfold/fold output width exceeds usize")
}

fn shader_source<T: WgpuScalar>(kind: KernelKind) -> String {
    let body = match kind {
        KernelKind::Unfold1d => {
            r#"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let out_l = output_layout.shape[2];
                let channels = output_layout.shape[1];
                let total = output_layout.shape[0] * channels * out_l;
                if (idx >= total) { return; }

                let l = idx % out_l;
                let channel_kernel = (idx / out_l) % channels;
                let n = idx / (out_l * channels);
                let channel = channel_kernel / params[0];
                let kernel = channel_kernel % params[0];
                let source = i32(l) * i32(params[1])
                    + i32(kernel) * i32(params[3]) - i32(params[2]);
                let output_index = index3(output_layout, n, channel_kernel, l);
                if (source >= 0 && u32(source) < input_layout.shape[2]) {
                    output[output_index] = input[index3(input_layout, n, channel, u32(source))];
                } else {
                    output[output_index] = {ZERO};
                }
            }
            "#
        }
        KernelKind::Fold1d => {
            r#"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let out_l = output_layout.shape[2];
                let channels = output_layout.shape[1];
                let total = output_layout.shape[0] * channels * out_l;
                if (idx >= total) { return; }

                let l = idx % out_l;
                let channel = (idx / out_l) % channels;
                let n = idx / (out_l * channels);
                var sum: {TYPE} = {ZERO};
                for (var kernel: u32 = 0u; kernel < params[0]; kernel = kernel + 1u) {
                    let numerator = i32(l) + i32(params[2])
                        - i32(kernel) * i32(params[3]);
                    if (numerator >= 0 && numerator % i32(params[1]) == 0) {
                        let source_l = u32(numerator / i32(params[1]));
                        if (source_l < input_layout.shape[2]) {
                            let channel_kernel = channel * params[0] + kernel;
                            sum = sum + input[index3(input_layout, n, channel_kernel, source_l)];
                        }
                    }
                }
                output[index3(output_layout, n, channel, l)] = sum;
            }
            "#
        }
        KernelKind::Unfold2d => {
            r#"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let locations = output_layout.shape[2];
                let channels = output_layout.shape[1];
                let total = output_layout.shape[0] * channels * locations;
                if (idx >= total) { return; }

                let location = idx % locations;
                let channel_kernel = (idx / locations) % channels;
                let n = idx / (locations * channels);
                let kernel_area = params[0] * params[1];
                let channel = channel_kernel / kernel_area;
                let kernel_offset = channel_kernel % kernel_area;
                let kh = kernel_offset / params[1];
                let kw = kernel_offset % params[1];
                let oh = location / params[8];
                let ow = location % params[8];
                let source_h = i32(oh) * i32(params[2])
                    + i32(kh) * i32(params[6]) - i32(params[4]);
                let source_w = i32(ow) * i32(params[3])
                    + i32(kw) * i32(params[7]) - i32(params[5]);
                let output_index = index3(output_layout, n, channel_kernel, location);
                if (source_h >= 0 && u32(source_h) < input_layout.shape[2]
                    && source_w >= 0 && u32(source_w) < input_layout.shape[3]) {
                    output[output_index] = input[index4(
                        input_layout, n, channel, u32(source_h), u32(source_w)
                    )];
                } else {
                    output[output_index] = {ZERO};
                }
            }
            "#
        }
        KernelKind::Fold2d => {
            r#"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let output_w = output_layout.shape[3];
                let output_h = output_layout.shape[2];
                let channels = output_layout.shape[1];
                let total = output_layout.shape[0] * channels * output_h * output_w;
                if (idx >= total) { return; }

                let w = idx % output_w;
                let h = (idx / output_w) % output_h;
                let channel = (idx / (output_w * output_h)) % channels;
                let n = idx / (output_w * output_h * channels);
                var sum: {TYPE} = {ZERO};
                for (var kh: u32 = 0u; kh < params[0]; kh = kh + 1u) {
                    let numerator_h = i32(h) + i32(params[4])
                        - i32(kh) * i32(params[6]);
                    if (numerator_h < 0 || numerator_h % i32(params[2]) != 0) {
                        continue;
                    }
                    let source_h = u32(numerator_h / i32(params[2]));
                    if (source_h >= input_layout.shape[2] / params[8]) {
                        continue;
                    }
                    for (var kw: u32 = 0u; kw < params[1]; kw = kw + 1u) {
                        let numerator_w = i32(w) + i32(params[5])
                            - i32(kw) * i32(params[7]);
                        if (numerator_w < 0 || numerator_w % i32(params[3]) != 0) {
                            continue;
                        }
                        let source_w = u32(numerator_w / i32(params[3]));
                        if (source_w >= params[8]) { continue; }
                        let location = source_h * params[8] + source_w;
                        if (location < input_layout.shape[2]) {
                            let channel_kernel = (channel * params[0] + kh) * params[1] + kw;
                            sum = sum + input[index3(
                                input_layout, n, channel_kernel, location
                            )];
                        }
                    }
                }
                output[index4(output_layout, n, channel, h, w)] = sum;
            }
            "#
        }
    };

    format!(
        r#"
        struct LayoutInfo {{
            offset: u32,
            ndim: u32,
            shape: array<u32, 8>,
            strides: array<u32, 8>,
        }}

        @group(0) @binding(0) var<storage, read> input: array<{TYPE}>;
        @group(0) @binding(1) var<storage, read_write> output: array<{TYPE}>;
        @group(0) @binding(2) var<storage, read> input_layout: LayoutInfo;
        @group(0) @binding(3) var<storage, read> output_layout: LayoutInfo;
        @group(0) @binding(4) var<storage, read> params: array<u32, 9>;

        fn index3(ly: LayoutInfo, i0: u32, i1: u32, i2: u32) -> u32 {{
            var index = ly.offset;
            if (ly.ndim > 0u) {{ index = index + i0 * ly.strides[0]; }}
            if (ly.ndim > 1u) {{ index = index + i1 * ly.strides[1]; }}
            if (ly.ndim > 2u) {{ index = index + i2 * ly.strides[2]; }}
            return index;
        }}

        fn index4(ly: LayoutInfo, i0: u32, i1: u32, i2: u32, i3: u32) -> u32 {{
            var index = ly.offset;
            if (ly.ndim > 0u) {{ index = index + i0 * ly.strides[0]; }}
            if (ly.ndim > 1u) {{ index = index + i1 * ly.strides[1]; }}
            if (ly.ndim > 2u) {{ index = index + i2 * ly.strides[2]; }}
            if (ly.ndim > 3u) {{ index = index + i3 * ly.strides[3]; }}
            return index;
        }}

        {body}
        "#,
        TYPE = T::WGSL_TYPE,
        body = body,
    )
    .replace("{TYPE}", T::WGSL_TYPE)
    .replace("{ZERO}", T::WGSL_ZERO)
}

#[allow(clippy::too_many_arguments)]
fn dispatch<T: WgpuScalar>(
    kind: KernelKind,
    input: &wgpu::Buffer,
    input_layout: &Layout,
    output: &wgpu::Buffer,
    output_layout: &Layout,
    params: [u32; 9],
) {
    let total = output_layout.shape().iter().product::<usize>();
    if total == 0 {
        return;
    }

    let ctx = crate::backend::get_wgpu_context();
    let input_layout_gpu = GpuLayoutInfo::from_layout(input_layout);
    let output_layout_gpu = GpuLayoutInfo::from_layout(output_layout);
    let input_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let output_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let params_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue
        .write_buffer(&input_layout_buf, 0, bytemuck::bytes_of(&input_layout_gpu));
    ctx.queue.write_buffer(
        &output_layout_buf,
        0,
        bytemuck::bytes_of(&output_layout_gpu),
    );
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

    let shader_src = shader_source::<T>(kind);
    let kind_name = match kind {
        KernelKind::Unfold1d => "unfold1d",
        KernelKind::Fold1d => "fold1d",
        KernelKind::Unfold2d => "unfold2d",
        KernelKind::Fold2d => "fold2d",
    };
    let key = format!("{kind_name}_{}", T::WGSL_TYPE);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unfold-fold-bind-group"),
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
                resource: input_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_layout_buf.as_entire_binding(),
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
            label: Some("unfold-fold-encoder"),
        });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("unfold-fold-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            u32::try_from(total.div_ceil(WORKGROUP_SIZE))
                .expect("unfold/fold dispatch exceeds the WGSL u32 workgroup range"),
            1,
            1,
        );
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// Dispatch one-dimensional sliding-window extraction on the device.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_unfold1d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &wgpu::Buffer,
    output_layout: &Layout,
) {
    let params = [
        parameter(kernel_size, "kernel_size"),
        parameter(stride, "stride"),
        parameter(padding, "padding"),
        parameter(dilation, "dilation"),
        0,
        0,
        0,
        0,
        parameter(
            output_width(input_layout, kernel_size, stride, padding, dilation),
            "output_width",
        ),
    ];
    dispatch::<T>(
        KernelKind::Unfold1d,
        input,
        input_layout,
        output,
        output_layout,
        params,
    );
}

/// Dispatch one-dimensional adjoint fold accumulation on the device.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fold1d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &wgpu::Buffer,
    output_layout: &Layout,
) {
    let params = [
        parameter(kernel_size, "kernel_size"),
        parameter(stride, "stride"),
        parameter(padding, "padding"),
        parameter(dilation, "dilation"),
        0,
        0,
        0,
        0,
        0,
    ];
    dispatch::<T>(
        KernelKind::Fold1d,
        input,
        input_layout,
        output,
        output_layout,
        params,
    );
}

/// Dispatch two-dimensional sliding-window extraction on the device.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_unfold2d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &Layout,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &wgpu::Buffer,
    output_layout: &Layout,
) {
    let params = [
        parameter(kernel_h, "kernel_h"),
        parameter(kernel_w, "kernel_w"),
        parameter(stride_h, "stride_h"),
        parameter(stride_w, "stride_w"),
        parameter(padding_h, "padding_h"),
        parameter(padding_w, "padding_w"),
        parameter(dilation_h, "dilation_h"),
        parameter(dilation_w, "dilation_w"),
        parameter(
            output_width(input_layout, kernel_w, stride_w, padding_w, dilation_w),
            "output_width",
        ),
    ];
    dispatch::<T>(
        KernelKind::Unfold2d,
        input,
        input_layout,
        output,
        output_layout,
        params,
    );
}

/// Dispatch two-dimensional adjoint fold accumulation on the device.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fold2d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &Layout,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &wgpu::Buffer,
    output_layout: &Layout,
) {
    let params = [
        parameter(kernel_h, "kernel_h"),
        parameter(kernel_w, "kernel_w"),
        parameter(stride_h, "stride_h"),
        parameter(stride_w, "stride_w"),
        parameter(padding_h, "padding_h"),
        parameter(padding_w, "padding_w"),
        parameter(dilation_h, "dilation_h"),
        parameter(dilation_w, "dilation_w"),
        parameter(
            output_width(
                &Layout::new([output_h, output_w].into()),
                kernel_w,
                stride_w,
                padding_w,
                dilation_w,
            ),
            "output_width",
        ),
    ];
    dispatch::<T>(
        KernelKind::Fold2d,
        input,
        input_layout,
        output,
        output_layout,
        params,
    );
}
