use super::cache::PIPELINE_CACHE;
use super::layout::{GpuLayoutError, GpuLayoutInfo};
use crate::backend::WgpuScalar;

/// Dispatch a WGSL shader for matrix multiplication.
pub fn dispatch_matmul<T: WgpuScalar>(
    a: &wgpu::Buffer,
    a_layout: &coeus_core::Layout,
    b: &wgpu::Buffer,
    b_layout: &coeus_core::Layout,
    c: &wgpu::Buffer,
    c_layout: &coeus_core::Layout,
) -> Result<(), GpuLayoutError> {
    let a_layout_gpu = GpuLayoutInfo::try_from_layout(a_layout)?;
    let b_layout_gpu = GpuLayoutInfo::try_from_layout(b_layout)?;
    let c_layout_gpu = GpuLayoutInfo::try_from_layout(c_layout)?;
    let ctx = crate::backend::get_wgpu_context();
    let wgsl_type = T::WGSL_TYPE;

    let a_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let b_layout_buf = crate::backend::PooledMetadataBuffer::new();
    let c_layout_buf = crate::backend::PooledMetadataBuffer::new();

    ctx.queue
        .write_buffer(&a_layout_buf, 0, bytemuck::bytes_of(&a_layout_gpu));
    ctx.queue
        .write_buffer(&b_layout_buf, 0, bytemuck::bytes_of(&b_layout_gpu));
    ctx.queue
        .write_buffer(&c_layout_buf, 0, bytemuck::bytes_of(&c_layout_gpu));

    let shader_src = format!(
        r#"
        struct LayoutInfo {{
            offset: u32,
            ndim: u32,
            shape: array<u32, 8>,
            strides: array<u32, 8>,
        }}

        @group(0) @binding(0) var<storage, read> a: array<{0}>;
        @group(0) @binding(1) var<storage, read> b: array<{0}>;
        @group(0) @binding(2) var<storage, read_write> c: array<{0}>;
        @group(0) @binding(3) var<storage, read> a_layout: LayoutInfo;
        @group(0) @binding(4) var<storage, read> b_layout: LayoutInfo;
        @group(0) @binding(5) var<storage, read> c_layout: LayoutInfo;

        var<workgroup> A_shared: array<array<{0}, 16>, 16>;
        var<workgroup> B_shared: array<array<{0}, 16>, 16>;

        @compute @workgroup_size(16, 16)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_id) local_id: vec3<u32>
        ) {{
            let row = global_id.y;
            let col = global_id.x;
            let local_row = local_id.y;
            let local_col = local_id.x;

            let m = a_layout.shape[0];
            let k = a_layout.shape[1];
            let n = b_layout.shape[1];

            let stride_a_row = a_layout.strides[0];
            let stride_a_col = a_layout.strides[1];
            let stride_b_row = b_layout.strides[0];
            let stride_b_col = b_layout.strides[1];

            var sum = {0}(0.0);
            let num_tiles = (k + 15u) / 16u;

            for (var tile_idx: u32 = 0u; tile_idx < num_tiles; tile_idx = tile_idx + 1u) {{
                // 1. Load A element into shared memory
                let col_a = tile_idx * 16u + local_col;
                if (row < m && col_a < k) {{
                    let offset_a = a_layout.offset + row * stride_a_row + col_a * stride_a_col;
                    A_shared[local_row][local_col] = a[offset_a];
                }} else {{
                    A_shared[local_row][local_col] = {0}(0.0);
                }}

                // 2. Load B element into shared memory
                let row_b = tile_idx * 16u + local_row;
                if (row_b < k && col < n) {{
                    let offset_b = b_layout.offset + row_b * stride_b_row + col * stride_b_col;
                    B_shared[local_row][local_col] = b[offset_b];
                }} else {{
                    B_shared[local_row][local_col] = {0}(0.0);
                }}

                // Synchronize to ensure all threads have finished loading the current tile
                workgroupBarrier();

                // 3. Accumulate product of the current tile
                for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
                    sum = sum + A_shared[local_row][i] * B_shared[i][local_col];
                }}

                // Synchronize before loading the next tile
                workgroupBarrier();
            }}

            if (row < m && col < n) {{
                let stride_c_row = c_layout.strides[0];
                let stride_c_col = c_layout.strides[1];
                let offset_c = c_layout.offset + row * stride_c_row + col * stride_c_col;
                c[offset_c] = sum;
            }}
        }}
        "#,
        wgsl_type
    );

    let key = format!("matmul_tiled_{}", wgsl_type);
    let pipeline = PIPELINE_CACHE.get_or_create(&key, &ctx.device, &shader_src, "main");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: a_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_layout_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: c_layout_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul-encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul-compute-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let m = a_layout.shape()[0];
        let n = b_layout.shape()[1];
        let workgroups_x = n.div_ceil(16);
        let workgroups_y = m.div_ceil(16);
        compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    Ok(())
}
