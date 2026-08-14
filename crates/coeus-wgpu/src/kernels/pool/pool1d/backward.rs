use super::super::validation::{parameter, try_layout};
use super::shader::{shader_source, PoolKind};
use crate::backend::{checked_numel, checked_workgroup_count, WgpuBackendError, WgpuScalar};
use crate::kernels::cache::PIPELINE_CACHE;
use coeus_core::Layout;

#[allow(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
fn dispatch_max_backward<T: WgpuScalar>(
    grad_out: &wgpu::Buffer,
    grad_out_layout: &Layout,
    input: &wgpu::Buffer,
    input_layout: &Layout,
    grad_input: &wgpu::Buffer,
    grad_input_layout: &Layout,
    params: [u32; 4],
) -> Result<(), WgpuBackendError> {
    let grad_out_layout_gpu = try_layout("max_pool1d_backward", grad_out_layout, 3)?;
    let input_layout_gpu = try_layout("max_pool1d_backward", input_layout, 3)?;
    let grad_input_layout_gpu = try_layout("max_pool1d_backward", grad_input_layout, 3)?;
    let total = checked_numel("max_pool1d_backward", grad_input_layout.shape())?;
    if total == 0 {
        return Ok(());
    }
    let workgroups = checked_workgroup_count("max_pool1d_backward", total)?;
    let ctx = crate::backend::get_wgpu_context();
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
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[allow(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
fn dispatch_avg_backward<T: WgpuScalar>(
    grad_out: &wgpu::Buffer,
    grad_out_layout: &Layout,
    grad_input: &wgpu::Buffer,
    grad_input_layout: &Layout,
    params: [u32; 4],
) -> Result<(), WgpuBackendError> {
    let grad_out_layout_gpu = try_layout("avg_pool1d_backward", grad_out_layout, 3)?;
    let grad_input_layout_gpu = try_layout("avg_pool1d_backward", grad_input_layout, 3)?;
    let total = checked_numel("avg_pool1d_backward", grad_input_layout.shape())?;
    if total == 0 {
        return Ok(());
    }
    let workgroups = checked_workgroup_count("avg_pool1d_backward", total)?;
    let ctx = crate::backend::get_wgpu_context();
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
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
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
) -> Result<(), WgpuBackendError> {
    dispatch_max_backward::<T>(
        grad_out,
        grad_out_layout,
        input,
        input_layout,
        grad_input,
        grad_input_layout,
        [
            parameter(kernel_size, "kernel_size")?,
            parameter(stride, "stride")?,
            parameter(padding, "padding")?,
            parameter(dilation, "dilation")?,
        ],
    )
}

#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub fn dispatch_avg_pool1d_backward<T: WgpuScalar>(
    grad_out: &wgpu::Buffer,
    grad_out_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &wgpu::Buffer,
    grad_input_layout: &Layout,
) -> Result<(), WgpuBackendError> {
    dispatch_avg_backward::<T>(
        grad_out,
        grad_out_layout,
        grad_input,
        grad_input_layout,
        [
            parameter(kernel_size, "kernel_size")?,
            parameter(stride, "stride")?,
            parameter(padding, "padding")?,
            parameter(dilation, "dilation")?,
        ],
    )
}
