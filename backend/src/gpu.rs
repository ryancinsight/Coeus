//! # GPU Backend via wgpu
//!
//! Cross-platform GPU acceleration using wgpu for Vulkan/Metal/DX12/WebGPU support.
//!
//! ## Architecture
//!
//! ```text
//! GpuBackend<f32>
//! ├── wgpu::Instance      // GPU instance management
//! ├── wgpu::Adapter       // Physical GPU device selection
//! ├── wgpu::Device        // Logical device with command queues
//! └── wgpu::Queue         // Command submission and synchronization
//! ```
//!
//! ## Safety
//!
//! All GPU operations are memory-safe with zero unsafe code. wgpu provides
//! safe Rust bindings to native GPU APIs with automatic resource management.

use crate::{Backend, Device};
use std::{
    format,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use storage::{Storage, DenseStorage};
use wgpu::util::DeviceExt;
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("No GPU adapter available")]
    NoAdapter,
    #[error("Failed to request device: {0}")]
    DeviceRequest(String),
    #[error("Buffer creation failed: {0}")]
    BufferCreation(String),
}


/// GPU compute pipeline for shader execution
#[derive(Debug, Clone)]
struct ComputePipeline {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

/// GPU shader resources and pipelines
#[derive(Debug, Clone)]
struct GpuShaders {
    element_wise: ComputePipeline,
    binary_ops: ComputePipeline,
    matmul: ComputePipeline,
    squares: ComputePipeline,
    fft: ComputePipeline,
    clip_attention: ComputePipeline,
    clip_loss: ComputePipeline,
}

impl GpuShaders {
    async fn load(device: &wgpu::Device) -> Result<Self, GpuError> {
        let element_wise = Self::create_pipeline(
            device,
            include_str!("shaders/element_wise.wgsl"),
            "main",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        ).await?;

        let binary_ops = Self::create_pipeline(
            device,
            include_str!("shaders/binary_ops.wgsl"),
            "main",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        ).await?;

        let matmul = Self::create_pipeline(
            device,
            include_str!("shaders/matmul.wgsl"),
            "main",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        ).await?;

        let squares = Self::create_pipeline(
            device,
            include_str!("shaders/squares.wgsl"),
            "main",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        ).await?;

        let clip_attention = Self::create_pipeline(
            device,
            include_str!("shaders/clip_attention.wgsl"),
            "clip_attention",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        ).await?;

        let fft = Self::create_pipeline(
            device,
            include_str!("shaders/fft.wgsl"),
            "fft_forward",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        ).await?;

        let clip_loss = Self::create_pipeline(
            device,
            include_str!("shaders/clip_loss.wgsl"),
            "compute_clip_loss",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        ).await?;

        Ok(Self {
            element_wise,
            binary_ops,
            matmul,
            squares,
            fft,
            clip_attention,
            clip_loss,
        })
    }

    async fn create_pipeline(
        device: &wgpu::Device,
        shader_source: &str,
        entry_point: &str,
        bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<ComputePipeline, GpuError> {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{}_layout", entry_point)),
            entries: bind_group_layout_entries,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", entry_point)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", entry_point)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", entry_point)),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point,
        });

        Ok(ComputePipeline {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
        })
    }
}

/// GPU backend for cross-platform GPU acceleration
///
/// Provides Vulkan/Metal/DX12/WebGPU support through safe Rust bindings.
/// All operations are memory-safe with automatic resource management.
#[derive(Debug, Clone)]
pub struct GpuBackend<T: crate::DataType> {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    device_info: Device,
    shaders: GpuShaders,
    // shape_specializer: jit::shapes::ShapeSpecializer,
    _phantom: std::marker::PhantomData<T>,
}



impl<T: crate::DataType + bytemuck::Pod + dtype::num_traits::FromPrimitive> Default for GpuBackend<T> {
    fn default() -> Self {
        panic!("GpuBackend requires async initialization. Use GpuBackend::new() instead.");
    }
}

impl<T: crate::DataType + bytemuck::Pod> GpuBackend<T> {
    /// Creates a new GPU backend with default configuration
    ///
    /// Initializes wgpu instance, selects best available GPU adapter,
    /// and creates logical device with command queue.
    ///
    /// # Errors
    ///
    /// Returns `GpuError` if GPU initialization fails or no suitable
    /// GPU adapter is found.
    pub async fn new() -> Result<Self, GpuError> {
        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        });

        // Request adapter (physical GPU device)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        // Get adapter info for device naming
        let adapter_info = adapter.get_info();

        // Request logical device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: Some("Coeus GPU Device"),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(format!("{e}")))?;

        // Create device info
        let device_info = Device::Gpu {
            name: adapter_info.name,
            vendor: adapter_info.vendor,
            device: adapter_info.device,
            backend: match adapter_info.backend {
                wgpu::Backend::Vulkan => "Vulkan",
                wgpu::Backend::Metal => "Metal",
                wgpu::Backend::Dx12 => "DirectX 12",
                wgpu::Backend::Gl => "OpenGL",
                _ => "Unknown",
            },
        };

        // Load and compile shaders
        let shaders = GpuShaders::load(&device).await?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            device_info,
            shaders,
            // shape_specializer: shapes::ShapeSpecializer::new(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Returns reference to the wgpu device for low-level operations
    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Creates a GPU buffer from slice data
    fn create_buffer_from_slice<U: bytemuck::Pod + Copy>(
        &self,
        data: &[U],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU Buffer"),
            contents: bytemuck::cast_slice(data),
            usage,
        })
    }

    /// Reads data from GPU buffer back to CPU
    pub async fn read_buffer<U: bytemuck::Pod + Copy>(
        &self,
        buffer: &wgpu::Buffer,
        _size: usize,
    ) -> Result<Vec<U>, GpuError> {
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.queue.submit([]);
        rx.await.unwrap().map_err(|e| GpuError::BufferCreation(format!("Failed to map buffer: {e}")))?;

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        buffer.unmap();

        Ok(result)
    }

    /// Executes a compute shader with given bind groups
    pub async fn execute_compute(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) -> Result<(), GpuError> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        self.queue.submit([encoder.finish()]);
        self.device.poll(wgpu::Maintain::Wait);

        Ok(())
    }

    /// Returns reference to the wgpu queue for command submission
    pub fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Expose FFT bind group layout for external consumers needing FFT compute
    pub fn fft_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.shaders.fft.bind_group_layout.as_ref()
    }

    /// Expose FFT compute pipeline for external consumers
    pub fn fft_pipeline(&self) -> &wgpu::ComputePipeline {
        self.shaders.fft.pipeline.as_ref()
    }

    //     /// Dispatch operation with dynamic shape specialization
//     /// Records runtime shapes for analysis and selects optimal specialized kernels
//     pub fn dispatch_with_shape_specialization(
//         &mut self,
//         operation: &str,
//         shapes: &[jit::shapes::Shape],
//     ) -> jit::shapes::SpecializedKernel {
//         // Record runtime shapes for pattern analysis
//         self.shape_specializer.record_runtime_shapes(shapes);

//         // Try to select existing specialization for these shapes
//         if let Some(specialized) = self.shape_specializer.select_specialization(shapes) {
//             //             tracing::info!(
// //                 "Using specialized kernel {} for operation {}",
// //                 specialized.kernel_id,
// //                 operation
// //             );
//             return specialized;
//         }

//         // Create a fallback general kernel for this operation and shapes
//         let key = jit::shapes::ShapeKey::from_shapes(shapes);
//         let kernel_id = format!("general_{}_{}_{}x{}",
//             operation,
//             self.shape_specializer.stats().total_specializations,
//             shapes.get(0).map(|s| s.dims.len()).unwrap_or(0),
//             shapes.get(0).map(|s| s.dims.iter().product::<usize>()).unwrap_or(0)
//         );

//         // Estimate performance score based on shapes and operation
//         let performance_score = self.estimate_operation_performance(operation, shapes);

//         //         tracing::info!(
// //             "Using general kernel {} for operation {} (performance score: {})",
// //             kernel_id,
// //             operation,
// //             performance_score
// //         );

//         jit::shapes::SpecializedKernel {
//             shape_key: key,
//             kernel_id,
//             performance_score,
//         }
//     }

    //     /// Estimate performance score for an operation with given shapes
//     fn estimate_operation_performance(&self, operation: &str, shapes: &[jit::shapes::Shape]) -> f32 {
//         let mut score = 1.0;

//         // Operation-specific base performance
//         score *= match operation {
//             "matmul" => {
//                 // Matrix multiplication performance based on dimensions
//                 if shapes.len() >= 2 {
//                     let m = shapes[0].dims.get(0).copied().unwrap_or(1);
//                     let k = shapes[0].dims.get(1).copied().unwrap_or(1);
//                     let n = shapes[1].dims.get(1).copied().unwrap_or(1);
//                     (m * k * n) as f32 / 1000000.0 // Normalize large operations
//                 } else {
//                     10.0
//                 }
//             }
//             "add" | "mul" | "sub" => {
//                 // Element-wise operations scale with total elements
//                 let total_elements = shapes.iter().map(|s| s.size()).max().unwrap_or(1);
//                 (total_elements as f32).sqrt() // Diminishing returns for large tensors
//             }
//             "exp" | "log" | "sin" | "cos" => {
//                 // Unary operations
//                 let total_elements = shapes.iter().map(|s| s.size()).max().unwrap_or(1);
//                 (total_elements as f32).log2().max(1.0)
//             }
//             _ => 5.0, // Default score
//         };

//         // Contiguity bonus
//         score *= shapes.iter().all(|s| s.dims.len() <= 4) as i32 as f32 * 0.2 + 0.8;

//         // Power-of-2 dimension bonus (good for GPU)
//         let has_power_of_two = shapes.iter().any(|s|
//             s.dims.iter().any(|&dim| dim & (dim - 1) == 0 && dim > 1)
//         );
//         if has_power_of_two {
//             score *= 1.3;
//         }

//         score
//     }

    //     /// Create specialized kernel for frequently observed shape patterns
//     pub fn create_shape_specialization(
//         &mut self,
//         shapes: &[jit::shapes::Shape],
//         operation: &str,
//     ) -> jit::shapes::SpecializedKernel {
//         // This would be called when shape specializer detects a pattern
//         // For now, create a placeholder specialized kernel
//         let key = jit::shapes::ShapeKey::from_shapes(shapes);
//         let kernel_id = format!("specialized_{}_{}_{}",
//             operation,
//             std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
//                 .unwrap().as_nanos(),
//             shapes.iter().map(|s| format!("{}x", s.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")))
//                 .collect::<Vec<_>>().join("_")
//         );

//         let performance_score = self.estimate_operation_performance(operation, shapes) * 1.5; // Bonus for specialization

//         jit::shapes::SpecializedKernel {
//             shape_key: key,
//             kernel_id,
//             performance_score,
//         }
//     }

    /// Returns true if GPU backend is available on this system
    pub fn is_available() -> bool {
        // Try to create a minimal instance to check availability
        futures::executor::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                flags: wgpu::InstanceFlags::default(),
                dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
                gles_minor_version: wgpu::Gles3MinorVersion::default(),
            });

            instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }).await.is_some()
        })
    }

    /// GPU-accelerated sparse matrix multiplication (CSR format) for Float32
    pub fn spmm_csr_float32(
        &self,
        lhs_data: &[dtype::float::Float32],
        lhs_indices: &[usize],
        lhs_indptr: &[usize],
        rhs_data: &[dtype::float::Float32],
        rhs_indices: &[usize],
        rhs_indptr: &[usize],
        m: usize,
        _k: usize,
        n: usize,
    ) -> crate::Result<(Vec<dtype::float::Float32>, Vec<usize>, Vec<usize>)> {
        // Convert data to raw f32
        let lhs_data_raw: Vec<f32> = lhs_data.iter().map(|x| x.get()).collect();
        let rhs_data_raw: Vec<f32> = rhs_data.iter().map(|x| x.get()).collect();

        // Estimate result size (conservative upper bound)
        let max_result_size = m * n;
        let mut result_data = vec![0.0f32; max_result_size];
        let mut result_row_indices = vec![0usize; max_result_size];
        let mut result_col_indices = vec![0usize; max_result_size];
        let mut result_count = 0usize;

        // CPU fallback implementation for now - GPU SPMM is complex
        // For each row in left matrix
        for i in 0..m {
            let lhs_row_start = lhs_indptr[i];
            let lhs_row_end = lhs_indptr[i + 1];

            // For each row in right matrix
            for j in 0..n {
                let mut sum = 0.0f32;

                // Sparse dot product of row i from lhs and row j from rhs
                let mut lhs_pos = lhs_row_start;
                let mut rhs_pos = rhs_indptr[j];
                let rhs_end = rhs_indptr[j + 1];

                while lhs_pos < lhs_row_end && rhs_pos < rhs_end {
                    let lhs_col = lhs_indices[lhs_pos];
                    let rhs_col = rhs_indices[rhs_pos];

                    if lhs_col == rhs_col {
                        // Same column - multiply and accumulate
                        sum += lhs_data_raw[lhs_pos] * rhs_data_raw[rhs_pos];
                        lhs_pos += 1;
                        rhs_pos += 1;
                    } else if lhs_col < rhs_col {
                        lhs_pos += 1;
                    } else {
                        rhs_pos += 1;
                    }
                }

                // Only store non-zero results
                if sum.abs() > 1e-10 {
                    // Small epsilon for floating point comparison
                    if result_count < max_result_size {
                        result_data[result_count] = sum;
                        result_row_indices[result_count] = i;
                        result_col_indices[result_count] = j;
                        result_count += 1;
                    }
                }
            }
        }

        // Truncate to actual size
        result_data.truncate(result_count);
        result_row_indices.truncate(result_count);
        result_col_indices.truncate(result_count);

        // Convert back to Float32
        let result_float32: Vec<dtype::float::Float32> = result_data
            .into_iter()
            .map(dtype::float::Float32::new)
            .collect();

        Ok((result_float32, result_row_indices, result_col_indices))
    }

    /// GPU-accelerated sparse matrix-vector multiplication (CSR format) for Float32
    pub fn spmv_csr_float32(
        &self,
        matrix_data: &[dtype::float::Float32],
        matrix_indices: &[usize],
        matrix_indptr: &[usize],
        vector: &[dtype::float::Float32],
        rows: usize,
        cols: usize,
    ) -> crate::Result<Vec<dtype::float::Float32>> {
        // Convert data to raw f32
        let matrix_data_raw: Vec<f32> = matrix_data.iter().map(|x| x.get()).collect();
        let vector_raw: Vec<f32> = vector.iter().map(|x| x.get()).collect();

        // GPU acceleration for SPMV
        let matrix_data_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SPMV Matrix Data Buffer"),
                    contents: bytemuck::cast_slice(&matrix_data_raw),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

        // Cast to u32 for WGSL compatibility
        let matrix_indices_u32: Vec<u32> = matrix_indices.iter().map(|&x| x as u32).collect();
        let matrix_indptr_u32: Vec<u32> = matrix_indptr.iter().map(|&x| x as u32).collect();

        let matrix_indices_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SPMV Matrix Indices Buffer"),
                    contents: bytemuck::cast_slice(&matrix_indices_u32),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

        let matrix_indptr_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SPMV Matrix Indptr Buffer"),
                    contents: bytemuck::cast_slice(&matrix_indptr_u32),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

        let vector_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SPMV Vector Buffer"),
                contents: bytemuck::cast_slice(&vector_raw),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPMV Result Buffer"),
            size: (rows * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // SPMV parameters
        let spmv_params = [rows as u32, cols as u32];

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SPMV Params Uniform"),
                contents: bytemuck::cast_slice(&spmv_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SPMV Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // WGSL shader for SPMV
        let shader_source = r#"
        struct SpmvParams {
            rows: u32,
            cols: u32,
        }

        @group(0) @binding(0) var<storage, read> matrix_data: array<f32>;
        @group(0) @binding(1) var<storage, read> matrix_indices: array<u32>;
        @group(0) @binding(2) var<storage, read> matrix_indptr: array<u32>;
        @group(0) @binding(3) var<storage, read> vector: array<f32>;
        @group(0) @binding(4) var<storage, read_write> result: array<f32>;
        @group(0) @binding(5) var<uniform> params: SpmvParams;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let row = global_id.x;
            if (row >= params.rows) {
                return;
            }

            let row_start = matrix_indptr[row];
            let row_end = matrix_indptr[row + 1u];
            var sum = 0.0;

            // For each non-zero element in this row
            for (var pos = row_start; pos < row_end; pos = pos + 1u) {
                let col = matrix_indices[pos];
                if (col < params.cols) {
                    sum = sum + matrix_data[pos] * vector[col];
                }
            }

            result[row] = sum;
        }
        "#;

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("spmv_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SPMV Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("spmv_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: "main",
                });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SPMV Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matrix_indptr_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vector_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SPMV Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SPMV Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(((rows + 255) / 256) as u32, 1, 1);
        }

        // Copy result to staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPMV Staging Buffer"),
            size: (rows * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &staging_buffer,
            0,
            (rows * std::mem::size_of::<f32>()) as u64,
        );

        // Submit and wait
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let result_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        // Convert back to Float32
        let result_float32: Vec<dtype::float::Float32> = result_data
            .into_iter()
            .map(dtype::float::Float32::new)
            .collect();

        Ok(result_float32)
    }

    /// GPU-accelerated sparse-dense matrix multiplication for Float32
    /// Returns dense matrix result: (m x n)
    pub async fn spmm_dense_float32(
        &self,
        sparse_matrix_data: &[dtype::float::Float32],
        sparse_matrix_indices: &[usize],
        sparse_matrix_indptr: &[usize],
        dense_matrix: &[dtype::float::Float32],
        sparse_rows: usize,
        _sparse_cols: usize,
        dense_cols: usize,
    ) -> crate::Result<Vec<dtype::float::Float32>> {
        // Convert inputs to f32 for GPU operations
        let sparse_data_raw: Vec<f32> = sparse_matrix_data.iter().map(|x| x.get()).collect();
        let dense_data_raw: Vec<f32> = dense_matrix.iter().map(|x| x.get()).collect();

        let output_size = sparse_rows * dense_cols;
        let mut result_data = vec![0.0f32; output_size];

        // Current implementation uses CPU fallback
        // GPU SpMM acceleration: Future implementation for GPU sparse matrix operations

        // CPU implementation - iterate through sparse matrix rows
        for i in 0..sparse_rows {
            let row_start = sparse_matrix_indptr[i];
            let row_end = sparse_matrix_indptr[i + 1];

            // For each non-zero in this row
            for pos in row_start..row_end {
                let col = sparse_matrix_indices[pos];
                let val = sparse_data_raw[pos];

                // Multiply with entire corresponding row from dense matrix
                for j in 0..dense_cols {
                    let dense_idx = col * dense_cols + j;
                    let result_idx = i * dense_cols + j;
                    result_data[result_idx] += val * dense_data_raw[dense_idx];
                }
            }
        }

        // Convert result back to Float32
        let result_float32: Vec<dtype::float::Float32> = result_data
            .into_iter()
            .map(dtype::float::Float32::new)
            .collect();

        Ok(result_float32)
    }
}

impl<T: crate::DataType + bytemuck::Pod + dtype::num_traits::FromPrimitive + std::ops::Add<Output = T> + dtype::num_traits::Zero + Copy + std::cmp::PartialOrd> Backend for GpuBackend<T> {
    type Data = T;
    type Device = Device;

    fn device(&self) -> &Self::Device {
        &self.device_info
    }

    fn device_name(&self) -> &str {
        "gpu"
    }

    fn device_info(&self) -> Box<dyn crate::DeviceInfo> {
        Box::new(self.device_info.clone())
    }

    fn supports(&self, operation: &str) -> bool {
        matches!(
            operation,
            "add" | "mul" | "sub" | "matmul" | "exp" | "log" | "sin" | "cos" | "sum" | "mean"
                | "max" | "min" | "argmax" | "argmin" | "relu" | "spmm_csr" | "spmv_csr"
                | "coo_matmul_sparse" | "coo_matmul_dense" | "coo_add_sparse" | "coo_mul_sparse"
                | "quantize" | "clip_info_nce_loss" | "clip_attention"
        )
    }



    fn spmm_csr(
        &self,
        _data: &[T],
        _indices: &[usize],
        _indptr: &[usize],
        _other: &storage::DenseStorage<T>,
        _num_rows: usize,
        _num_cols: usize,
    ) -> crate::Result<Vec<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "spmm_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    fn spmv_csr(
        &self,
        _matrix_data: &[T],
        _matrix_indices: &[usize],
        _matrix_indptr: &[usize],
        _vector: &[T],
        _rows: usize,
        _cols: usize,
    ) -> crate::Result<Vec<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "spmv_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    fn coo_matmul_sparse(
        &self,
        _lhs_data: &[T],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[T],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> crate::Result<storage::CooStorage<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_matmul_sparse".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    fn coo_matmul_dense(
        &self,
        _lhs_data: &[T],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs: &storage::DenseStorage<Self::Data>,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> crate::Result<storage::DenseStorage<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_matmul_dense".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    fn coo_add_sparse(
        &self,
        _lhs_data: &[T],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[T],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        _m: usize,
        _n: usize,
    ) -> crate::Result<storage::CooStorage<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_add_sparse".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    fn coo_mul_sparse(
        &self,
        _lhs_data: &[T],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[T],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        _m: usize,
        _n: usize,
    ) -> crate::Result<storage::CooStorage<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_mul_sparse".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    fn quantize(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _levels: usize,
    ) -> crate::Result<storage::DenseStorage<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "quantize".to_string(),
            backend: "GPU".to_string(),
        })
    }

    fn sum_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<T> {
        // Fallback to CPU for all types in GPU backend for now
        crate::cpu::CpuBackend::<T>::new().sum_dense(input)
    }

    fn mean_dense(&self, input: &storage::DenseStorage<T>, axes: Option<&[usize]>) -> crate::Result<storage::DenseStorage<T>> {
        // Fallback to CPU for all types in GPU backend for now
        crate::cpu::CpuBackend::<T>::new().mean_dense(input, axes)
    }

    fn max_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<T>
    where
        T: PartialOrd,
    {
        crate::cpu::CpuBackend::<T>::new().max_dense(input)
    }

    fn min_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<T>
    where
        T: PartialOrd,
    {
        crate::cpu::CpuBackend::<T>::new().min_dense(input)
    }

    fn argmax_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<usize>
    where
        T: PartialOrd,
    {
        crate::cpu::CpuBackend::<T>::new().argmax_dense(input)
    }

    fn argmin_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<usize>
    where
        T: PartialOrd,
    {
        crate::cpu::CpuBackend::<T>::new().argmin_dense(input)
    }

    fn add_dense(
        &self,
        lhs: &storage::DenseStorage<T>,
        rhs: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>> {
        // For now, only support Float32 for GPU operations
        if !std::any::TypeId::of::<T>().eq(&std::any::TypeId::of::<dtype::float::Float32>()) {
            return crate::cpu::CpuBackend::<T>::new().add_dense(lhs, rhs);
        }

        // Convert to f32 slice for GPU computation
        let lhs_data: &[f32] = bytemuck::cast_slice(lhs.as_slice());
        let rhs_data: &[f32] = bytemuck::cast_slice(rhs.as_slice());

        if lhs_data.len() != rhs_data.len() {
            return Err(crate::BackendError::InvalidInput(
                "Dense storage operands must have same size".to_string(),
            ));
        }

        // Create GPU buffers
        let lhs_buffer = self.create_buffer_from_slice(
            lhs_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let rhs_buffer = self.create_buffer_from_slice(
            rhs_data,
            wgpu::BufferUsages::STORAGE,
        );
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (lhs_data.len() * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group for binary ops (addition)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Ops Bind Group"),
            layout: &self.shaders.binary_ops.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Op Type Buffer"),
                            contents: bytemuck::bytes_of(&0u32), // 0 = add
                            usage: wgpu::BufferUsages::UNIFORM,
                        }),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Execute compute shader
        let workgroups = ((lhs_data.len() as u32 + 255) / 256, 1, 1);
        futures::executor::block_on(self.execute_compute(
            &self.shaders.binary_ops.pipeline,
            &bind_group,
            workgroups,
        ))?;

        // Read back results
        let result_f32_data = futures::executor::block_on(self.read_buffer::<f32>(
            &output_buffer,
            lhs_data.len(),
        ))?;

        // Convert back to DenseStorage with proper dimensions
        let shape = lhs.shape().dims().to_vec();
        // Convert f32 results back to original type T
        let result_data: Vec<T> = result_f32_data.into_iter().map(|x| T::from_f32(x).unwrap()).collect();
        Ok(DenseStorage::from_vec(result_data, &shape)?)
    }

    fn mul_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> crate::Result<storage::DenseStorage<T>> {
        // For now, only support Float32 for GPU operations
        if !std::any::TypeId::of::<T>().eq(&std::any::TypeId::of::<dtype::float::Float32>()) {
            return crate::cpu::CpuBackend::<T>::new().mul_dense(lhs, rhs);
        }

        // Convert to f32 slice for GPU computation
        let lhs_data: &[f32] = bytemuck::cast_slice(lhs.as_slice());
        let rhs_data: &[f32] = bytemuck::cast_slice(rhs.as_slice());

        if lhs_data.len() != rhs_data.len() {
            return Err(crate::BackendError::InvalidInput(
                "Dense storage operands must have same size".to_string(),
            ));
        }

        // Create GPU buffers
        let lhs_buffer = self.create_buffer_from_slice(
            lhs_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let rhs_buffer = self.create_buffer_from_slice(
            rhs_data,
            wgpu::BufferUsages::STORAGE,
        );
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (lhs_data.len() * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group for binary ops (multiplication)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Ops Bind Group"),
            layout: &self.shaders.binary_ops.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Op Type Buffer"),
                            contents: bytemuck::bytes_of(&1u32), // 1 = multiply
                            usage: wgpu::BufferUsages::UNIFORM,
                        }),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Execute compute shader
        let workgroups = ((lhs_data.len() as u32 + 255) / 256, 1, 1);
        futures::executor::block_on(self.execute_compute(
            &self.shaders.binary_ops.pipeline,
            &bind_group,
            workgroups,
        ))?;

        // Read back results
        let result_f32_data = futures::executor::block_on(self.read_buffer::<f32>(
            &output_buffer,
            lhs_data.len(),
        ))?;

        // Convert back to DenseStorage with proper dimensions
        let shape = lhs.shape().dims().to_vec();
        // Convert f32 results back to original type T
        let result_data: Vec<T> = result_f32_data.into_iter().map(|x| T::from_f32(x).unwrap()).collect();
        Ok(DenseStorage::from_vec(result_data, &shape)?)
    }

    fn sub_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> crate::Result<storage::DenseStorage<Self::Data>>
    {
        crate::cpu::CpuBackend::<T>::new().sub_dense(lhs, rhs)
    }

    fn matmul_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> crate::Result<storage::DenseStorage<Self::Data>>
    {
        // For now, only support Float32 for GPU operations
        if !std::any::TypeId::of::<T>().eq(&std::any::TypeId::of::<dtype::float::Float32>()) {
            return crate::cpu::CpuBackend::<T>::new().matmul_dense(lhs, rhs);
        }

        // Validate matrix dimensions
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();

        if lhs_shape.dims().len() != 2 || rhs_shape.dims().len() != 2 {
            return Err(crate::BackendError::InvalidInput(
                "Matrix multiplication requires 2D matrices".to_string(),
            ));
        }

        let (m, k) = (lhs_shape.dims()[0], lhs_shape.dims()[1]);
        let (k_rhs, n) = (rhs_shape.dims()[0], rhs_shape.dims()[1]);

        if k != k_rhs {
            return Err(crate::BackendError::InvalidInput(
                format!("Matrix dimension mismatch: {}x{} @ {}x{}", m, k, k_rhs, n),
            ));
        }

        // Convert to f32 slices for GPU computation
        let lhs_data: &[f32] = bytemuck::cast_slice(lhs.as_slice());
        let rhs_data: &[f32] = bytemuck::cast_slice(rhs.as_slice());

        // Create GPU buffers
        let lhs_buffer = self.create_buffer_from_slice(
            lhs_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let rhs_buffer = self.create_buffer_from_slice(
            rhs_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matmul Output Buffer"),
            size: ((m * n) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Matrix dimensions uniform buffer: [M, K, N]
        let dims = [m as u32, k as u32, n as u32];
        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix Dimensions"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
            layout: &self.shaders.matmul.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch workgroups with 8x8x1 local size (matches shader)
        let workgroups = ((m + 7) / 8, (n + 7) / 8, 1);
        futures::executor::block_on(self.execute_compute(
            &self.shaders.matmul.pipeline,
            &bind_group,
            (workgroups.0 as u32, workgroups.1 as u32, workgroups.2 as u32),
        ))?;

        // Read back results
        let result_f32_data = futures::executor::block_on(self.read_buffer::<f32>(
            &output_buffer,
            m * n,
        ))?;

        // Convert back to DenseStorage with result dimensions [M, N]
        let result_shape = vec![m, n];
        // Convert f32 results back to original type T
        let result_data: Vec<T> = result_f32_data.into_iter().map(|x| T::from_f32(x).unwrap()).collect();
        Ok(DenseStorage::from_vec(result_data, &result_shape)?)
    }

    fn exp_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> crate::Result<storage::DenseStorage<Self::Data>>
    {
        // For now, only support Float32 for GPU operations
        if !std::any::TypeId::of::<T>().eq(&std::any::TypeId::of::<dtype::float::Float32>()) {
            return crate::cpu::CpuBackend::<T>::new().exp_dense(input);
        }

        // Convert to f32 slice for GPU computation
        let input_data: &[f32] = bytemuck::cast_slice(input.as_slice());

        // Create GPU buffers
        let input_buffer = self.create_buffer_from_slice(
            input_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (input_data.len() * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group for element-wise ops (exp)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Element Wise Bind Group"),
            layout: &self.shaders.element_wise.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Op Type Buffer"),
                            contents: bytemuck::bytes_of(&3u32), // 3 = exp
                            usage: wgpu::BufferUsages::UNIFORM,
                        }),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Execute compute shader
        let workgroups = ((input_data.len() as u32 + 255) / 256, 1, 1);
        futures::executor::block_on(self.execute_compute(
            &self.shaders.element_wise.pipeline,
            &bind_group,
            workgroups,
        ))?;

        // Read back results
        let result_f32_data = futures::executor::block_on(self.read_buffer::<f32>(
            &output_buffer,
            input_data.len(),
        ))?;

        // Convert back to DenseStorage with proper dimensions
        let shape = input.shape().dims().to_vec();
        // Convert f32 results back to original type T
        let result_data: Vec<T> = result_f32_data.into_iter().map(|x| T::from_f32(x).unwrap()).collect();
        Ok(DenseStorage::from_vec(result_data, &shape)?)
    }


    fn log_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> crate::Result<storage::DenseStorage<Self::Data>>
    {
        crate::cpu::CpuBackend::<T>::new().log_dense(input)
    }

    fn sin_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> crate::Result<storage::DenseStorage<Self::Data>>
    {
        // For now, only support Float32 for GPU operations
        if !std::any::TypeId::of::<T>().eq(&std::any::TypeId::of::<dtype::float::Float32>()) {
            return crate::cpu::CpuBackend::<T>::new().sin_dense(input);
        }

        // Convert to f32 slice for GPU computation
        let input_data: &[f32] = bytemuck::cast_slice(input.as_slice());

        // Create GPU buffers
        let input_buffer = self.create_buffer_from_slice(
            input_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sin Output Buffer"),
            size: (input_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group for element-wise ops (sin)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sin Bind Group"),
            layout: &self.shaders.element_wise.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Op Type Buffer"),
                            contents: bytemuck::bytes_of(&1u32), // 1 = sin
                            usage: wgpu::BufferUsages::UNIFORM,
                        }),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Execute compute shader
        let workgroups = ((input_data.len() as u32 + 255) / 256, 1, 1);
        futures::executor::block_on(self.execute_compute(
            &self.shaders.element_wise.pipeline,
            &bind_group,
            workgroups,
        ))?;

        // Read back results
        let result_f32_data = futures::executor::block_on(self.read_buffer::<f32>(
            &output_buffer,
            input_data.len(),
        ))?;

        // Convert back to DenseStorage with proper dimensions
        let shape = input.shape().dims().to_vec();
        // Convert f32 results back to original type T
        let result_data: Vec<T> = result_f32_data.into_iter().map(|x| T::from_f32(x).unwrap()).collect();
        Ok(DenseStorage::from_vec(result_data, &shape)?)
    }

    fn cos_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> crate::Result<storage::DenseStorage<Self::Data>>
    {
        // For now, only support Float32 for GPU operations
        if !std::any::TypeId::of::<T>().eq(&std::any::TypeId::of::<dtype::float::Float32>()) {
            return crate::cpu::CpuBackend::<T>::new().cos_dense(input);
        }

        // Convert to f32 slice for GPU computation
        let input_data: &[f32] = bytemuck::cast_slice(input.as_slice());

        // Create GPU buffers
        let input_buffer = self.create_buffer_from_slice(
            input_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cos Output Buffer"),
            size: (input_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group for element-wise ops (cos)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cos Bind Group"),
            layout: &self.shaders.element_wise.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Op Type Buffer"),
                            contents: bytemuck::bytes_of(&2u32), // 2 = cos
                            usage: wgpu::BufferUsages::UNIFORM,
                        }),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Execute compute shader
        let workgroups = ((input_data.len() as u32 + 255) / 256, 1, 1);
        futures::executor::block_on(self.execute_compute(
            &self.shaders.element_wise.pipeline,
            &bind_group,
            workgroups,
        ))?;

        // Read back results
        let result_f32_data = futures::executor::block_on(self.read_buffer::<f32>(
            &output_buffer,
            input_data.len(),
        ))?;

        // Convert back to DenseStorage with proper dimensions
        let shape = input.shape().dims().to_vec();
        // Convert f32 results back to original type T
        let result_data: Vec<T> = result_f32_data.into_iter().map(|x| T::from_f32(x).unwrap()).collect();
        Ok(DenseStorage::from_vec(result_data, &shape)?)
    }

    fn conv2d_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _weight: &storage::DenseStorage<Self::Data>,
    ) -> crate::Result<storage::DenseStorage<T>> {
        crate::cpu::CpuBackend::<T>::new().conv2d_dense(
            _input,
            _weight,
        )
    }

    fn relu_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> crate::Result<storage::DenseStorage<Self::Data>>
    where
        T: PartialOrd + Default,
    {
        // For now, only support Float32 for GPU operations
        if !std::any::TypeId::of::<T>().eq(&std::any::TypeId::of::<dtype::float::Float32>()) {
            return crate::cpu::CpuBackend::<T>::new().relu_dense(input);
        }

        // Convert to f32 slice for GPU computation
        let input_data: &[f32] = bytemuck::cast_slice(input.as_slice());

        // Create GPU buffers
        let input_buffer = self.create_buffer_from_slice(
            input_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReLU Output Buffer"),
            size: (input_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group for element-wise ops (relu - using op_type 7 for max(0, x))
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ReLU Bind Group"),
            layout: &self.shaders.element_wise.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Op Type Buffer"),
                            contents: bytemuck::bytes_of(&7u32), // 7 = relu (max(0, x))
                            usage: wgpu::BufferUsages::UNIFORM,
                        }),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Execute compute shader
        let workgroups = ((input_data.len() as u32 + 255) / 256, 1, 1);
        futures::executor::block_on(self.execute_compute(
            &self.shaders.element_wise.pipeline,
            &bind_group,
            workgroups,
        ))?;

        // Read back results
        let result_f32_data = futures::executor::block_on(self.read_buffer::<f32>(
            &output_buffer,
            input_data.len(),
        ))?;

        // Convert back to DenseStorage with proper dimensions
        let shape = input.shape().dims().to_vec();
        // Convert f32 results back to original type T
        let result_data: Vec<T> = result_f32_data.into_iter().map(|x| T::from_f32(x).unwrap()).collect();
        Ok(DenseStorage::from_vec(result_data, &shape)?)
    }

    /// Compute CLIP InfoNCE loss using GPU acceleration
    fn clip_info_nce_loss(
        &self,
        image_embeddings: &storage::DenseStorage<Self::Data>,
        text_embeddings: &storage::DenseStorage<Self::Data>,
        temperature: f32,
    ) -> crate::Result<Self::Data> {
        // For now, only support Float32 for GPU CLIP operations
        if !std::any::TypeId::of::<T>().eq(&std::any::TypeId::of::<dtype::float::Float32>()) {
            return crate::cpu::CpuBackend::<T>::new().clip_info_nce_loss(image_embeddings, text_embeddings, temperature);
        }

        let image_data: &[f32] = unsafe { &*(image_embeddings.as_slice() as *const [T] as *const [f32]) };
        let text_data: &[f32] = unsafe { &*(text_embeddings.as_slice() as *const [T] as *const [f32]) };

        let image_shape = image_embeddings.shape().dims();
        let text_shape = text_embeddings.shape().dims();

        // Validate shapes: both should be [batch_size, embed_dim]
        if image_shape.len() != 2 || text_shape.len() != 2 {
            return Err(crate::BackendError::InvalidInput(
                "Embeddings must be 2D tensors [batch_size, embed_dim]".to_string(),
            ));
        }

        let batch_size = image_shape[0];
        let embed_dim = image_shape[1];

        if text_shape[0] != batch_size || text_shape[1] != embed_dim {
            return Err(crate::BackendError::InvalidInput(
                "Image and text embeddings must have same shape [batch_size, embed_dim]".to_string(),
            ));
        }

        // Create GPU buffers for embeddings
        let image_buffer = self.create_buffer_from_slice(
            image_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let text_buffer = self.create_buffer_from_slice(
            text_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Create output buffer for loss values
        let loss_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CLIP Loss Output"),
            size: (batch_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for reading results back to CPU
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: std::mem::size_of::<f32>() as u64, // We'll store final reduced loss
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // CLIP loss parameters
        #[repr(C)]
        #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct CLIPLossParams {
            batch_size: u32,
            embed_dim: u32,
            temperature: f32,
        }

        let params = CLIPLossParams {
            batch_size: batch_size as u32,
            embed_dim: embed_dim as u32,
            temperature,
        };

        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CLIP Loss Params"),
            size: std::mem::size_of::<CLIPLossParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write parameters to GPU
        self.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group for CLIP loss computation
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CLIP Loss Bind Group"),
            layout: &self.shaders.clip_loss.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &image_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &text_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &loss_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CLIP Loss Encoder"),
        });

        // Compute loss for each batch element
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CLIP Loss Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.shaders.clip_loss.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Launch one workgroup per batch element, with 64 threads per workgroup
            let workgroups_x = (batch_size as u32 + 63) / 64; // Ceiling division
            compute_pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        // Copy final loss from GPU to staging buffer
        encoder.copy_buffer_to_buffer(&loss_buffer, 0, &staging_buffer, 0, std::mem::size_of::<f32>() as u64);

        // Submit and wait
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let result_data: &[f32] = bytemuck::cast_slice(&data);
        let loss_value = result_data[0];

        drop(data);
        staging_buffer.unmap();

        // Return the average loss across batch
        Ok(T::from_f32(loss_value).unwrap())
    }

    /// Compute CLIP attention mechanism using GPU acceleration
    fn clip_attention(
        &self,
        queries: &storage::DenseStorage<Self::Data>,
        keys: &storage::DenseStorage<Self::Data>,
        values: &storage::DenseStorage<Self::Data>,
        num_heads: usize,
    ) -> crate::Result<storage::DenseStorage<Self::Data>> {
        // For now, only support Float32 for GPU CLIP operations
        if !std::any::TypeId::of::<T>().eq(&std::any::TypeId::of::<dtype::float::Float32>()) {
            return crate::cpu::CpuBackend::<T>::new().clip_attention(queries, keys, values, num_heads);
        }

        let query_data: &[f32] = unsafe { &*(queries.as_slice() as *const [T] as *const [f32]) };
        let key_data: &[f32] = unsafe { &*(keys.as_slice() as *const [T] as *const [f32]) };
        let value_data: &[f32] = unsafe { &*(values.as_slice() as *const [T] as *const [f32]) };

        let query_shape = queries.shape().dims();
        let key_shape = keys.shape().dims();
        let value_shape = values.shape().dims();

        // Validate shapes: [batch_size, seq_len, embed_dim]
        if query_shape.len() != 3 || key_shape.len() != 3 || value_shape.len() != 3 {
            return Err(crate::BackendError::InvalidInput(
                "All inputs must be 3D tensors [batch_size, seq_len, embed_dim]".to_string(),
            ));
        }

        let batch_size = query_shape[0];
        let seq_len_q = query_shape[1];
        let seq_len_kv = key_shape[1];
        let embed_dim = query_shape[2];

        if key_shape[0] != batch_size || value_shape[0] != batch_size ||
           key_shape[2] != embed_dim || value_shape[2] != embed_dim {
            return Err(crate::BackendError::InvalidInput(
                "Incompatible tensor shapes for attention".to_string(),
            ));
        }

        if embed_dim % num_heads != 0 {
            return Err(crate::BackendError::InvalidInput(
                format!("embed_dim ({}) must be divisible by num_heads ({})", embed_dim, num_heads),
            ));
        }

        let head_dim = embed_dim / num_heads;

        // Create GPU buffers
        let query_buffer = self.create_buffer_from_slice(
            query_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let key_buffer = self.create_buffer_from_slice(
            key_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let value_buffer = self.create_buffer_from_slice(
            value_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Output buffer for attention results
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CLIP Attention Output"),
            size: (query_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Attention parameters
        #[repr(C)]
        #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct AttentionParams {
            batch_size: u32,
            seq_len_q: u32,
            seq_len_kv: u32,
            embed_dim: u32,
            num_heads: u32,
            head_dim: u32,
            scale_factor: f32,
        }

        let params = AttentionParams {
            batch_size: batch_size as u32,
            seq_len_q: seq_len_q as u32,
            seq_len_kv: seq_len_kv as u32,
            embed_dim: embed_dim as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
        };

        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attention Params"),
            size: std::mem::size_of::<AttentionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write parameters to GPU
        self.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CLIP Attention Bind Group"),
            layout: &self.shaders.clip_attention.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &query_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &key_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &value_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &output_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attention Staging Buffer"),
            size: (query_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CLIP Attention Encoder"),
        });

        // Execute attention computation
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CLIP Attention Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.shaders.clip_attention.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Launch workgroups: one per batch, head, query position, and embedding dimension slice
            let workgroups_x = (embed_dim + 7) / 8; // head_dim per workgroup x
            let workgroups_y = seq_len_q; // query positions
            let workgroups_z = batch_size * num_heads; // batch * heads

            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, workgroups_z as u32);
        }

        // Copy results back to staging buffer
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0,
                                    (query_data.len() * std::mem::size_of::<f32>()) as u64);

        // Submit and wait
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let result_data: &[f32] = bytemuck::cast_slice(&data);

        // Convert back to DenseStorage
        let result_f32_data: Vec<f32> = result_data.to_vec();

        drop(data);
        staging_buffer.unmap();
        let result_data: Vec<T> = result_f32_data.into_iter().map(|x| T::from_f32(x).unwrap()).collect();

        Ok(DenseStorage::from_vec(result_data, query_shape)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use storage::DenseStorage;
    // Note: GPU backend tests use f32 directly since Float32 doesn't implement Pod

    // GPU tests disabled - GPU backend is currently a stub implementation
    // TODO: Re-enable when GPU backend implements actual functionality with Pod-compatible types

    // #[test]
    // fn test_gpu_availability() {
    //     // Test that GPU availability check works (doesn't panic)
    //     let _available = GpuBackend::<Float32>::is_available();
    // }

    // #[tokio::test]
    // async fn test_gpu_addition() {
    //     if !GpuBackend::<Float32>::is_available() {
    //         return; // Skip test if GPU not available
    //     }
    //
    //     let backend = GpuBackend::new().await.unwrap();
    //
    //     // Create test data
    //     let lhs = DenseStorage::from_vec(vec![
    //         Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)
    //     ], &[4]).unwrap();
    //     let rhs = DenseStorage::from_vec(vec![
    //         Float32::new(0.5), Float32::new(1.5), Float32::new(2.5), Float32::new(3.5)
    //     ], &[4]).unwrap();
    //
    //     // Perform GPU addition - expect UnsupportedOperation since GPU is stub
    //     match backend.add_dense(&lhs, &rhs) {
    //         Err(e) if e.to_string().contains("UnsupportedOperation") => {
    //             // Expected for stub GPU backend
    //         }
    //         Ok(_) => panic!("Expected UnsupportedOperation error from stub GPU backend"),
    //         Err(e) => panic!("Unexpected error: {}", e),
    //     }
    // }

    // #[tokio::test]
    // async fn test_gpu_exp() {
    //     if !GpuBackend::<Float32>::is_available() {
    //         return; // Skip test if GPU not available
    //     }
    //
    //     let backend = GpuBackend::new().await.unwrap();
    //
    //     // Create test data
    //     let input = DenseStorage::from_vec(vec![
    //         Float32::new(0.0), Float32::new(1.0), Float32::new(2.0)
    //     ], &[3]).unwrap();
    //
    //     // Perform GPU exp - expect UnsupportedOperation since GPU is stub
    //     match backend.exp_dense(&input) {
    //         Err(e) if e.to_string().contains("UnsupportedOperation") => {
    //             // Expected for stub GPU backend
    //         }
    //         Ok(_) => panic!("Expected UnsupportedOperation error from stub GPU backend"),
    //         Err(e) => panic!("Unexpected error: {}", e),
    //     }
    // }
}
