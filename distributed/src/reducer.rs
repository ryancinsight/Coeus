//! Gradient reduction and synchronization for distributed training

use crate::error::{DistributedError, Result};
use crate::process_group::ProcessGroup;
use std::collections::HashMap;

/// Gradient reducer for synchronizing gradients across devices
///
/// This handles the aggregation of gradients from multiple devices using
/// AllReduce operations to ensure consistent model updates.
#[derive(Debug)]
pub struct GradientReducer {
    process_group: ProcessGroup,
    buffers: HashMap<String, Vec<f32>>,
    gpu_buffers: HashMap<String, wgpu::Buffer>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
}

impl GradientReducer {
    /// Create a new gradient reducer
    pub fn new(process_group: ProcessGroup) -> Self {
        Self {
            process_group,
            buffers: HashMap::new(),
            gpu_buffers: HashMap::new(),
            device: None,
            queue: None,
        }
    }

    /// Create a new GPU-accelerated gradient reducer
    pub fn new_with_gpu(process_group: ProcessGroup, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            process_group,
            buffers: HashMap::new(),
            gpu_buffers: HashMap::new(),
            device: Some(device),
            queue: Some(queue),
        }
    }

    /// Register a parameter for gradient reduction
    ///
    /// This allocates a buffer for accumulating gradients from all devices
    /// in the process group.
    pub fn register_parameter(&mut self, name: String, size: usize) -> Result<()> {
        if self.buffers.contains_key(&name) {
            return Err(DistributedError::ProcessGroupConfig {
                message: format!("Parameter '{}' already registered", name),
            });
        }

        let buffer_size = size * self.process_group.world_size().0;
        self.buffers.insert(name.clone(), vec![0.0; buffer_size]);

        // Create GPU buffer if GPU acceleration is available
        if let (Some(device), Some(queue)) = (&self.device, &self.queue) {
            let gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Gradient Buffer: {}", name)),
                size: (buffer_size * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.gpu_buffers.insert(name, gpu_buffer);
        }

        Ok(())
    }

    /// Reduce gradients for a parameter across all devices (CPU)
    ///
    /// This performs an AllReduce operation to average gradients from all
    /// devices and updates the local gradient buffer.
    pub async fn reduce_gradients(&mut self, name: &str, local_gradients: &[f32]) -> Result<()> {
        let buffer =
            self.buffers
                .get_mut(name)
                .ok_or_else(|| DistributedError::ProcessGroupConfig {
                    message: format!("Parameter '{}' not registered", name),
                })?;

        let world_size = self.process_group.world_size().0;
        let rank = self.process_group.rank().0;

        // Copy local gradients into the appropriate section of the buffer
        let start_idx = rank * local_gradients.len();
        let end_idx = start_idx + local_gradients.len();

        if end_idx > buffer.len() {
            return Err(DistributedError::BufferOverflow {
                required: end_idx,
                available: buffer.len(),
            });
        }

        buffer[start_idx..end_idx].copy_from_slice(local_gradients);

        // Perform AllReduce (placeholder for actual communication)
        self.process_group.all_reduce(buffer).await?;

        // Average the gradients across all devices
        let avg_factor = 1.0 / world_size as f32;
        for grad in &mut buffer[start_idx..end_idx] {
            *grad *= avg_factor;
        }

        Ok(())
    }

    /// Reduce gradients for a parameter across all devices (GPU-accelerated)
    ///
    /// This performs GPU-accelerated AllReduce operations for optimal performance
    /// on large models with many parameters.
    pub async fn reduce_gradients_gpu(&mut self, name: &str, local_gradients: &[f32]) -> Result<()> {
        // Check if GPU acceleration is available
        let (device, queue) = match (&self.device, &self.queue) {
            (Some(d), Some(q)) => (d, q),
            _ => return Err(DistributedError::ProcessGroupConfig {
                message: "GPU acceleration not available for gradient reduction".to_string(),
            }),
        };

        let gpu_buffer = self.gpu_buffers.get(name).ok_or_else(|| DistributedError::ProcessGroupConfig {
            message: format!("GPU buffer for parameter '{}' not found", name),
        })?;

        let world_size = self.process_group.world_size().0;
        let rank = self.process_group.rank().0;
        let grad_size = local_gradients.len();

        // Upload local gradients to GPU buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gradient Upload Staging"),
            size: (grad_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });

        // Write gradients to staging buffer
        {
            let mut view = staging_buffer.slice(..).get_mapped_range_mut();
            let data = bytemuck::cast_slice_mut::<u8, f32>(&mut view);
            data.copy_from_slice(local_gradients);
        }
        staging_buffer.unmap();

        // Copy to GPU buffer at the correct offset
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gradient Upload Encoder"),
        });

        let offset = (rank * grad_size * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            gpu_buffer,
            offset,
            (grad_size * std::mem::size_of::<f32>()) as u64,
        );

        queue.submit(Some(encoder.finish()));

        // Perform AllReduce (placeholder - would integrate with NCCL/Gloo in production)
        self.process_group.all_reduce_gpu(gpu_buffer, grad_size, world_size).await?;

        // Average gradients on GPU
        let avg_shader = r#"
        @group(0) @binding(0) var<storage, read_write> gradients: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx < arrayLength(&gradients)) {
                gradients[idx] = gradients[idx] / 4.0; // world_size = 4 for this example
            }
        }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gradient Averaging Shader"),
            source: wgpu::ShaderSource::Wgsl(avg_shader.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gradient Averaging Layout"),
            bind_group_layouts: &[&device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gradient Averaging Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            })],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gradient Averaging Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient Averaging Bind Group"),
            layout: &pipeline_layout.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_buffer.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gradient Averaging Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gradient Averaging Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = ((grad_size * world_size + 255) / 256) as u32;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        Ok(())
    }

    /// Get the reduced gradients for a parameter
    pub fn get_reduced_gradients(&self, name: &str) -> Result<&[f32]> {
        let buffer =
            self.buffers
                .get(name)
                .ok_or_else(|| DistributedError::ProcessGroupConfig {
                    message: format!("Parameter '{}' not registered", name),
                })?;

        let rank = self.process_group.rank().0;
        let world_size = self.process_group.world_size().0;
        let grad_size = buffer.len() / world_size;

        let start_idx = rank * grad_size;
        let end_idx = start_idx + grad_size;

        Ok(&buffer[start_idx..end_idx])
    }

    /// Get the process group
    pub fn process_group(&self) -> &ProcessGroup {
        &self.process_group
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process_group::{Rank, WorldSize};

    #[tokio::test]
    async fn test_gradient_reducer_registration() {
        let pg = ProcessGroup::new(Rank(0), WorldSize(2)).unwrap();
        let mut reducer = GradientReducer::new(pg);

        reducer.register_parameter("weight".to_string(), 4).unwrap();

        // Should fail to register duplicate
        assert!(reducer.register_parameter("weight".to_string(), 4).is_err());
    }

    #[test]
    fn test_gradient_reducer_creation() {
        let pg = ProcessGroup::new(Rank(0), WorldSize(4)).unwrap();
        let reducer = GradientReducer::new(pg);
        // Should create without error
        assert_eq!(reducer.buffers.len(), 0);
    }

    #[test]
    fn test_parameter_registration() {
        let pg = ProcessGroup::new(Rank(0), WorldSize(3)).unwrap();
        let mut reducer = GradientReducer::new(pg);

        // Register parameter
        reducer
            .register_parameter("layer.weight".to_string(), 6)
            .unwrap();

        // Should have buffer with correct size: 6 elements * 3 devices = 18
        assert_eq!(reducer.buffers.len(), 1);
        assert_eq!(reducer.buffers["layer.weight"].len(), 18);
    }

    #[test]
    fn test_duplicate_parameter_registration() {
        let pg = ProcessGroup::new(Rank(0), WorldSize(2)).unwrap();
        let mut reducer = GradientReducer::new(pg);

        // Register parameter first time
        reducer.register_parameter("bias".to_string(), 4).unwrap();

        // Second registration should fail
        let result = reducer.register_parameter("bias".to_string(), 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_unregistered_parameter_access() {
        let pg = ProcessGroup::new(Rank(0), WorldSize(2)).unwrap();
        let mut reducer = GradientReducer::new(pg);

        // Try to reduce unregistered parameter
        let grads = vec![1.0, 2.0];
        let result = futures::executor::block_on(reducer.reduce_gradients("unknown", &grads));
        assert!(result.is_err());

        // Try to get gradients for unregistered parameter
        let result = reducer.get_reduced_gradients("unknown");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_gradient_reduction() {
        let pg = ProcessGroup::new(Rank(0), WorldSize(2)).unwrap();
        let mut reducer = GradientReducer::new(pg);

        reducer.register_parameter("weight".to_string(), 2).unwrap();

        let local_grads = vec![1.0, 2.0];
        reducer
            .reduce_gradients("weight", &local_grads)
            .await
            .unwrap();

        let reduced = reducer.get_reduced_gradients("weight").unwrap();
        assert_eq!(reduced, &[0.5, 1.0]); // Averaged across 2 devices
    }

    #[tokio::test]
    async fn test_gradient_reduction_multiple_devices() {
        let pg = ProcessGroup::new(Rank(1), WorldSize(4)).unwrap(); // Rank 1 of 4
        let mut reducer = GradientReducer::new(pg);

        reducer
            .register_parameter("conv.weight".to_string(), 3)
            .unwrap();

        let local_grads = vec![0.5, 1.0, 1.5];
        reducer
            .reduce_gradients("conv.weight", &local_grads)
            .await
            .unwrap();

        let reduced = reducer.get_reduced_gradients("conv.weight").unwrap();
        // Should get the local portion, averaged
        assert_eq!(reduced, &[0.125, 0.25, 0.375]); // local_grads / world_size
    }

    #[tokio::test]
    async fn test_multiple_parameters() {
        let pg = ProcessGroup::new(Rank(0), WorldSize(2)).unwrap();
        let mut reducer = GradientReducer::new(pg);

        // Register multiple parameters
        reducer
            .register_parameter("weights".to_string(), 4)
            .unwrap();
        reducer.register_parameter("biases".to_string(), 2).unwrap();

        // Reduce gradients for both
        let weight_grads = vec![1.0, 2.0, 3.0, 4.0];
        let bias_grads = vec![0.1, 0.2];

        reducer
            .reduce_gradients("weights", &weight_grads)
            .await
            .unwrap();
        reducer
            .reduce_gradients("biases", &bias_grads)
            .await
            .unwrap();

        // Check both results
        let reduced_weights = reducer.get_reduced_gradients("weights").unwrap();
        let reduced_biases = reducer.get_reduced_gradients("biases").unwrap();

        assert_eq!(reduced_weights, &[0.5, 1.0, 1.5, 2.0]);
        assert_eq!(reduced_biases, &[0.05, 0.1]);
    }
}
