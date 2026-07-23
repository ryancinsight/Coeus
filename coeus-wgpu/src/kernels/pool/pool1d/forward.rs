use super::shader::{parameter, shader_source, PoolKind, WORKGROUP_SIZE};
use crate::backend::WgpuScalar;
use crate::kernels::cache::PIPELINE_CACHE;
use crate::kernels::layout::GpuLayoutInfo;
use coeus_core::Layout;

fn dispatch_forward<T: WgpuScalar>(
    kind: PoolKind,
    input: &wgpu::Buffer,
    input_layout: &Layout,
    output: &wgpu::Buffer,
    output_layout: &Layout,
    params: [u32; 4],
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
        PoolKind::MaxForward => "max_pool1d",
        PoolKind::AvgForward => "avg_pool1d",
        PoolKind::MaxBackward | PoolKind::AvgBackward => {
            unreachable!("backward pool kinds do not use the forward dispatcher")
        }
    };
    let key = format!("{kind_name}_{}", T::WGSL_TYPE);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pool1d-bind-group"),
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
            label: Some("pool1d-encoder"),
        });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pool1d-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            u32::try_from(total.div_ceil(WORKGROUP_SIZE))
                .expect("pool1d dispatch exceeds the WGSL u32 workgroup range"),
            1,
            1,
        );
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_max_pool1d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &wgpu::Buffer,
    output_layout: &Layout,
) {
    dispatch_forward::<T>(
        PoolKind::MaxForward,
        input,
        input_layout,
        output,
        output_layout,
        [
            parameter(kernel_size, "kernel_size"),
            parameter(stride, "stride"),
            parameter(padding, "padding"),
            parameter(dilation, "dilation"),
        ],
    );
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_avg_pool1d<T: WgpuScalar>(
    input: &wgpu::Buffer,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &wgpu::Buffer,
    output_layout: &Layout,
) {
    dispatch_forward::<T>(
        PoolKind::AvgForward,
        input,
        input_layout,
        output,
        output_layout,
        [
            parameter(kernel_size, "kernel_size"),
            parameter(stride, "stride"),
            parameter(padding, "padding"),
            parameter(dilation, "dilation"),
        ],
    );
}
