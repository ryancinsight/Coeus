//! GPU-Accelerated Optimizer Backend
//!
//! Provides GPU shader acceleration for optimization algorithms with support for both
//! sparse and dense parameter updates. Integrates with the existing wgpu backend
//! to provide high-performance optimizer operations.

use crate::error::OptimError;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Errors that can occur in GPU optimizer backend operations
#[derive(Debug, thiserror::Error)]
pub enum GpuOptimizerError {
    #[error("Failed to create GPU device: {0}")]
    DeviceCreation(String),

    #[error("Failed to compile shader: {0}")]
    ShaderCompilation(String),

    #[error("Failed to create compute pipeline: {0}")]
    PipelineCreation(String),

    #[error("Buffer operation failed: {0}")]
    BufferError(String),

    #[error("GPU operation not supported: {0}")]
    UnsupportedOperation(String),
}

/// Placeholder for GPU backend configuration
/// In a real implementation, this would integrate with wgpu
#[derive(Debug, Clone)]
pub struct GpuOptimizerConfig {
    /// Sparsity threshold above which to use sparse kernels (default: 0.1 = 10%)
    pub sparsity_threshold: f64,
    /// Whether GPU acceleration is enabled
    pub gpu_enabled: bool,
    /// Maximum batch size for dense operations
    pub max_batch_size: usize,
}

impl Default for GpuOptimizerConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 0.1, // 10% sparsity threshold
            gpu_enabled: true,
            max_batch_size: 32,
        }
    }
}

/// GPU optimizer backend with wgpu resources for compute shader acceleration
#[derive(Debug)]
pub struct GpuOptimizerBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sparse_rmsprop_pipeline: wgpu::ComputePipeline,
    dense_rmsprop_pipeline: wgpu::ComputePipeline,
    sparse_bind_group_layout: wgpu::BindGroupLayout,
    dense_bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuOptimizerBackend {
    /// Create a new GPU optimizer backend with wgpu initialization
    pub async fn new() -> Result<Self, OptimError> {
        // Initialize wgpu instance and device
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| OptimError::BackendError {
                message: "No suitable GPU adapter found".into(),
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: Some("Coeus GPU Optimizer Device"),
                },
                None,
            )
            .await
            .map_err(|e| OptimError::BackendError {
                message: format!("Failed to request GPU device: {e}"),
            })?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Load RMSprop shader source
        let shader_source = include_str!("../../backend/src/shaders/sparse_dense_optimizers.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RMSprop Shaders"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create sparse RMSprop bind group layout
        let sparse_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sparse RMSprop Bind Group Layout"),
                entries: &[
                    // rmsprop_sparse_indices
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // rmsprop_gradients
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // rmsprop_parameters
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // square_avg
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // grad_avg
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // momentum_buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // rmsprop_config
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create dense RMSprop bind group layout
        let dense_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Dense RMSprop Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create compute pipeline layouts
        let sparse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sparse RMSprop Pipeline Layout"),
                bind_group_layouts: &[&sparse_bind_group_layout],
                push_constant_ranges: &[],
            });

        let dense_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Dense RMSprop Pipeline Layout"),
                bind_group_layouts: &[&dense_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create compute pipelines
        let sparse_rmsprop_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sparse RMSprop Pipeline"),
                layout: Some(&sparse_pipeline_layout),
                module: &shader,
                entry_point: "sparse_rmsprop_update",
            });

        let dense_rmsprop_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Dense RMSprop Pipeline"),
                layout: Some(&dense_pipeline_layout),
                module: &shader,
                entry_point: "dense_rmsprop_batch_update",
            });

        Ok(Self {
            device,
            queue,
            sparse_rmsprop_pipeline,
            dense_rmsprop_pipeline,
            sparse_bind_group_layout,
            dense_bind_group_layout,
        })
    }

    /// Analyze gradient sparsity to determine optimal dispatch strategy
    pub fn analyze_gradient_sparsity(grades: &[f64]) -> f64 {
        if grades.is_empty() {
            return 0.0;
        }

        let zero_elements = grades.iter().filter(|&&x| x.abs() < 1e-7).count();
        zero_elements as f64 / grades.len() as f64
    }

    /// Determine if gradients should be processed with sparse or dense kernels
    pub fn should_use_sparse_kernels(grades: &[f64], threshold: f64) -> bool {
        Self::analyze_gradient_sparsity(grades) >= threshold
    }

    /// Execute sparse RMSprop update on GPU using wgpu compute shader
    pub fn rmsprop_sparse_update(
        &self,
        params: &mut [f32],
        grads: &[f32],
        indices: &[u32],
        square_avg: &mut [f32],
        grad_avg: &mut [f32],
        momentum_buffer: &mut [f32],
        config: &RMSpropConfig,
    ) -> Result<(), OptimError> {
        if indices.len() != grads.len() {
            return Err(OptimError::BackendError {
                message: "Indices and gradients arrays must have same length".into(),
            });
        }

        // Create GPU buffers
        let indices_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RMSprop Sparse Indices"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let grads_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RMSprop Sparse Gradients"),
                contents: bytemuck::cast_slice(grads),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RMSprop Parameters"),
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let square_avg_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RMSprop Square Avg"),
                contents: bytemuck::cast_slice(square_avg),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let grad_avg_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RMSprop Grad Avg"),
                contents: bytemuck::cast_slice(grad_avg),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let momentum_buffer_gpu =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("RMSprop Momentum Buffer"),
                    contents: bytemuck::cast_slice(momentum_buffer),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });

        let config_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RMSprop Config"),
                contents: bytemuck::bytes_of(config),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparse RMSprop Bind Group"),
            layout: &self.sparse_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grads_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: square_avg_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grad_avg_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: momentum_buffer_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sparse RMSprop Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sparse RMSprop Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.sparse_rmsprop_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Workgroup size is 256 (defined in WGSL), dispatch enough workgroups to cover all indices
            let workgroups = ((indices.len() + 255) / 256) as u32;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Create staging buffers for reading results back
        #[allow(clippy::manual_slice_size_calculation)]
        let params_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Staging"),
            size: (params.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        #[allow(clippy::manual_slice_size_calculation)]
        let square_avg_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Square Avg Staging"),
            size: (square_avg.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        #[allow(clippy::manual_slice_size_calculation)]
        let grad_avg_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grad Avg Staging"),
            size: (grad_avg.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        #[allow(clippy::manual_slice_size_calculation)]
        let momentum_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Momentum Staging"),
            size: (momentum_buffer.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy results back to staging buffers
        #[allow(clippy::manual_slice_size_calculation)]
        let buffer_size = |slice: &[f32]| (slice.len() * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&params_buffer, 0, &params_staging, 0, buffer_size(params));
        encoder.copy_buffer_to_buffer(
            &square_avg_buffer,
            0,
            &square_avg_staging,
            0,
            buffer_size(square_avg),
        );
        encoder.copy_buffer_to_buffer(
            &grad_avg_buffer,
            0,
            &grad_avg_staging,
            0,
            buffer_size(grad_avg),
        );
        encoder.copy_buffer_to_buffer(
            &momentum_buffer_gpu,
            0,
            &momentum_staging,
            0,
            buffer_size(momentum_buffer),
        );

        self.queue.submit(Some(encoder.finish()));

        // Read results back synchronously (this will need to be made async in the future)
        Self::read_buffer_sync(&self.device, &params_staging, params)?;
        Self::read_buffer_sync(&self.device, &square_avg_staging, square_avg)?;
        Self::read_buffer_sync(&self.device, &grad_avg_staging, grad_avg)?;
        Self::read_buffer_sync(&self.device, &momentum_staging, momentum_buffer)?;

        Ok(())
    }

    /// Helper method to read GPU buffer back to CPU memory synchronously
    fn read_buffer_sync<T: bytemuck::Pod + Copy>(
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        output: &mut [T],
    ) -> Result<(), OptimError> {
        // This is a synchronous read for simplicity - in production this should be async
        let buffer_slice = buffer.slice(..);

        buffer_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap(); // For now, panic on error - should be handled properly
        });

        device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let slice: &[T] = bytemuck::cast_slice(&data);
        output.copy_from_slice(slice);

        drop(data);
        buffer.unmap();

        Ok(())
    }

    /// Execute dense RMSprop update on GPU using wgpu compute shader
    pub fn rmsprop_dense_update(
        &self,
        params: &mut [f32],
        grads: &[f32],
        square_avg: &mut [f32],
        grad_avg: &mut [f32],
        momentum_buffer: &mut [f32],
        batch_size: usize,
        config: &RMSpropConfig,
    ) -> Result<(), OptimError> {
        // Validate input dimensions
        if grads.len() != batch_size * params.len() {
            return Err(OptimError::BackendError {
                message: format!(
                    "Gradients length {} doesn't match batch_size {} * params length {}",
                    grads.len(),
                    batch_size,
                    params.len()
                ),
            });
        }

        // All parameter state arrays must be the same length as params
        if square_avg.len() != params.len()
            || grad_avg.len() != params.len()
            || momentum_buffer.len() != params.len()
        {
            return Err(OptimError::BackendError {
                message: "Parameter state arrays must have same length as parameters".into(),
            });
        }

        // Create GPU buffers
        let dummy_indices = [0u32];
        let indices_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dense RMSprop Dummy Indices"),
                contents: bytemuck::cast_slice(&dummy_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let grads_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dense RMSprop Gradients"),
                contents: bytemuck::cast_slice(grads),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dense RMSprop Parameters"),
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let square_avg_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dense RMSprop Square Avg"),
                contents: bytemuck::cast_slice(square_avg),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let grad_avg_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dense RMSprop Grad Avg"),
                contents: bytemuck::cast_slice(grad_avg),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let momentum_buffer_gpu =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Dense RMSprop Momentum Buffer"),
                    contents: bytemuck::cast_slice(momentum_buffer),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });

        // Create config with correct param_count for the kernel
        let mut kernel_config = *config;
        kernel_config.param_count = params.len() as u32;

        let config_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dense RMSprop Config"),
                contents: bytemuck::bytes_of(&kernel_config),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dense RMSprop Bind Group"),
            layout: &self.dense_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grads_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: square_avg_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grad_avg_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: momentum_buffer_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dense RMSprop Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dense RMSprop Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.dense_rmsprop_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Workgroup size is 256 (defined in WGSL), dispatch one workgroup per parameter
            let workgroups = ((params.len() + 255) / 256) as u32;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Create staging buffers for reading results back
        #[allow(clippy::manual_slice_size_calculation)]
        let params_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dense Params Staging"),
            size: (params.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        #[allow(clippy::manual_slice_size_calculation)]
        let square_avg_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dense Square Avg Staging"),
            size: (square_avg.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        #[allow(clippy::manual_slice_size_calculation)]
        let grad_avg_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dense Grad Avg Staging"),
            size: (grad_avg.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        #[allow(clippy::manual_slice_size_calculation)]
        let momentum_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dense Momentum Staging"),
            size: (momentum_buffer.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy results back to staging buffers
        #[allow(clippy::manual_slice_size_calculation)]
        {
            encoder.copy_buffer_to_buffer(
                &params_buffer,
                0,
                &params_staging,
                0,
                (params.len() * std::mem::size_of::<f32>()) as u64,
            );
            encoder.copy_buffer_to_buffer(
                &square_avg_buffer,
                0,
                &square_avg_staging,
                0,
                (square_avg.len() * std::mem::size_of::<f32>()) as u64,
            );
            encoder.copy_buffer_to_buffer(
                &grad_avg_buffer,
                0,
                &grad_avg_staging,
                0,
                (grad_avg.len() * std::mem::size_of::<f32>()) as u64,
            );
            encoder.copy_buffer_to_buffer(
                &momentum_buffer_gpu,
                0,
                &momentum_staging,
                0,
                (momentum_buffer.len() * std::mem::size_of::<f32>()) as u64,
            );
        }

        self.queue.submit(Some(encoder.finish()));

        // Read results back synchronously
        Self::read_buffer_sync(&self.device, &params_staging, params)?;
        Self::read_buffer_sync(&self.device, &square_avg_staging, square_avg)?;
        Self::read_buffer_sync(&self.device, &grad_avg_staging, grad_avg)?;
        Self::read_buffer_sync(&self.device, &momentum_staging, momentum_buffer)?;

        Ok(())
    }
}

/// Adam optimizer configuration for GPU kernels
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub step: u64,
    pub param_count: u32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            step: 1,
            param_count: 0,
        }
    }
}

/// RMSprop optimizer configuration for GPU kernels
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RMSpropConfig {
    pub lr: f32,
    pub alpha: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    pub centered: u32, // 1 if centered, 0 otherwise
    pub param_count: u32,
    pub _pad: u32,
}

impl Default for RMSpropConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: 0, // 0 = not centered
            param_count: 0,
            _pad: 0,
        }
    }
}

/// Placeholder trait for GPU-aware optimizers
/// Simplified to avoid complex trait bounds
pub trait GpuAcceleratedOptimizer {
    fn gpu_available(&self) -> bool {
        false
    }
    fn gpu_backend(&self) -> Option<&GpuOptimizerBackend> {
        None
    }
    fn gpu_config(&self) -> Option<&GpuOptimizerConfig> {
        None
    }
    fn set_gpu_config(&mut self, _config: GpuOptimizerConfig) {}
}
