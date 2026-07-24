use super::shader::{parameter, shader_source, PoolKind, WORKGROUP_SIZE};
use crate::backend::WgpuScalar;
use crate::kernels::cache::PIPELINE_CACHE;
use crate::kernels::layout::GpuLayoutInfo;
use coeus_core::Layout;

#[allow(clippy::too_many_arguments)]
fn dispatch_max_backward<T: WgpuScalar>(
    grad_out: &wgpu::Buffer,
    grad_out_layout: &Layout,
    input: &wgpu::Buffer,
    input_layout: &Layout,
    grad_input: &wgpu::Buffer,
    grad_input_layout: &Layout,
    params: [u32; 4],
) {
    let total = grad_input_layout.shape().iter().product::<usize>();
    if total == 0 {
        return;
    }
    let ctx = crate::backend::get_wgpu_context();
    let grad_out_layout_gpu = GpuLayoutInfo::from_layout(grad_out_layout);
    let input_layout_gpu = GpuLayoutInfo::from_layout(input_layout);
    let grad_input_layout_gpu = GpuLayoutInfo::from_layout(grad_input_layout);
    let grad_out_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let input_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let grad_input_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let params_buf = crate::backend::PooledMetadataBuffer::new();
    ctx.queue.write_buffer(
        &grad_out_layout_buf,
        0,
        bytemuck::bytes_of(&grad_out_layout_gpu),
    );
    ctx.queue
        .write_buffer(&input_layout_buf, 0, bytemuck::bytes_of(&input_layout_gpu));
    ctx.queue.write_buffer(
        &grad_input_layout_buf,
        0,
        bytemuck::bytes_of(&grad_input_layout_gpu),
    );
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));
    let shader_src = shader_source::<T>(PoolKind::MaxBackward);
    let pipeline = PIPELINE_CACHE.get_or_create(
        &format!("max_pool1d_backward_{}", T::WGSL_TYPE),
        &ctx.device,
        &shader_src,
        "main",
    );
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("max-pool1d-backward-bind-group"),
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
                resource: grad_out_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: input_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: grad_input_layout_buf.as_entire_binding(),
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
            label: Some("max-pool1d-backward-encoder"),
        });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("max-pool1d-backward-compute-pass"),
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
fn dispatch_avg_backward<T: WgpuScalar>(
    grad_out: &wgpu::Buffer,
    grad_out_layout: &Layout,
    grad_input: &wgpu::Buffer,
    grad_input_layout: &Layout,
    params: [u32; 4],
) {
    let total = grad_input_layout.shape().iter().product::<usize>();
    if total == 0 {
        return;
    }
    let ctx = crate::backend::get_wgpu_context();
    let grad_out_layout_gpu = GpuLayoutInfo::from_layout(grad_out_layout);
    let grad_input_layout_gpu = GpuLayoutInfo::from_layout(grad_input_layout);
    let grad_out_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let grad_input_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let params_buf = crate::backend::PooledMetadataBuffer::new();
    ctx.queue.write_buffer(
        &grad_out_layout_buf,
        0,
        bytemuck::bytes_of(&grad_out_layout_gpu),
    );
    ctx.queue.write_buffer(
        &grad_input_layout_buf,
        0,
        bytemuck::bytes_of(&grad_input_layout_gpu),
    );
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));
    let shader_src = shader_source::<T>(PoolKind::AvgBackward);
    let pipeline = PIPELINE_CACHE.get_or_create(
        &format!("avg_pool1d_backward_{}", T::WGSL_TYPE),
        &ctx.device,
        &shader_src,
        "main",
    );
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("avg-pool1d-backward-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: grad_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grad_input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grad_out_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grad_input_layout_buf.as_entire_binding(),
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
            label: Some("avg-pool1d-backward-encoder"),
        });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("avg-pool1d-backward-compute-pass"),
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
pub fn dispatch_max_pool1d_backward<T: WgpuScalar>(
    grad_out: &wgpu::Buffer,
    grad_out_layout: &Layout,
    input: &wgpu::Buffer,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &wgpu::Buffer,
    grad_input_layout: &Layout,
) {
    dispatch_max_backward::<T>(
        grad_out,
        grad_out_layout,
        input,
        input_layout,
        grad_input,
        grad_input_layout,
        [
            parameter(kernel_size, "kernel_size"),
            parameter(stride, "stride"),
            parameter(padding, "padding"),
            parameter(dilation, "dilation"),
        ],
    );
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_avg_pool1d_backward<T: WgpuScalar>(
    grad_out: &wgpu::Buffer,
    grad_out_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &wgpu::Buffer,
    grad_input_layout: &Layout,
) {
    dispatch_avg_backward::<T>(
        grad_out,
        grad_out_layout,
        grad_input,
        grad_input_layout,
        [
            parameter(kernel_size, "kernel_size"),
            parameter(stride, "stride"),
            parameter(padding, "padding"),
            parameter(dilation, "dilation"),
        ],
    );
}
