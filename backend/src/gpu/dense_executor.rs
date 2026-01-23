//! GPU Dense Kernel Executor
//!
//! Provides GPU-accelerated execution of dense tensor operations using WGPU compute shaders.
//! Supports f32 and f64 data types through type-specific shader pipelines.

use crate::Result;
use wgpu::util::DeviceExt;

/// WGSL shader source files embedded at compile time
pub mod shader_sources {
    /// Binary operations (add, mul)
    pub const BINARY_OPS: &str = include_str!("shaders/binary_ops.wgsl");
    /// Matrix multiplication
    pub const MATMUL: &str = include_str!("shaders/matmul.wgsl");
    /// Element-wise operations
    pub const ELEMENT_WISE: &str = include_str!("shaders/element_wise.wgsl");
    /// Sparse matrix-vector multiplication
    pub const SPMV: &str = include_str!("shaders/spmv.wgsl");
    /// Sparse kernels
    pub const SPARSE_KERNELS: &str = include_str!("shaders/sparse_kernels.wgsl");
    /// Conv2D kernel
    pub const CONV2D: &str = include_str!("shaders/conv2d.wgsl");
    /// Reduction kernel
    pub const REDUCTION: &str = include_str!("shaders/reduction.wgsl");
}

/// GPU executor for dense tensor operations
///
/// Provides high-performance GPU implementations for common tensor operations.
/// Uses WGPU for cross-platform GPU support (Vulkan, Metal, DX12, WebGPU).
#[derive(Debug)]
pub struct GpuDenseExecutor {
    /// WGPU device handle
    device: wgpu::Device,
    /// Command queue for GPU execution
    queue: wgpu::Queue,
    /// Binary operations pipeline (add, mul)
    binary_ops_pipeline: wgpu::ComputePipeline,
    /// Matrix multiplication pipeline
    matmul_pipeline: wgpu::ComputePipeline,
    /// SpMV pipeline for sparse-vector multiplication
    spmv_pipeline: wgpu::ComputePipeline,
    /// SpMM pipeline for sparse-matrix dense-matrix multiplication
    spmm_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for binary operations
    binary_ops_layout: wgpu::BindGroupLayout,
    /// Bind group layout for matmul
    matmul_layout: wgpu::BindGroupLayout,
    /// Bind group layout for spmv
    spmv_layout: wgpu::BindGroupLayout,
    /// Bind group layout for spmm
    spmm_layout: wgpu::BindGroupLayout,
    /// Pipeline for Conv2D
    conv2d_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for Conv2D
    conv2d_layout: wgpu::BindGroupLayout,
    /// Pipeline for Element-wise operations
    element_wise_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for Element-wise operations
    element_wise_layout: wgpu::BindGroupLayout,
    /// Pipeline for Reduction operations
    reduction_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for Reduction operations
    reduction_layout: wgpu::BindGroupLayout,
}

impl GpuDenseExecutor {
    /// Create a new GPU dense executor
    ///
    /// Initializes WGPU and compiles all required compute shaders.
    /// Returns None if no suitable GPU is available.
    pub async fn new() -> Result<Option<Self>> {
        // Request GPU adapter
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
            .await;

        let adapter = match adapter {
            Some(a) => a,
            None => return Ok(None),
        };

        // Request device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: Some("Coeus GPU Dense Executor"),
                },
                None,
            )
            .await
            .map_err(|e| crate::BackendError::GpuError(format!("Device creation failed: {}", e)))?;

        // Create bind group layouts
        let binary_ops_layout = Self::create_binary_ops_layout(&device);
        let matmul_layout = Self::create_matmul_layout(&device);
        let spmv_layout = Self::create_spmv_layout(&device);
        let spmm_layout = Self::create_spmm_layout(&device);
        let conv2d_layout = Self::create_conv2d_layout(&device);
        let element_wise_layout = Self::create_element_wise_layout(&device);
        let reduction_layout = Self::create_reduction_layout(&device);

        // Compile shaders
        let binary_ops_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Binary Ops Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_sources::BINARY_OPS.into()),
        });

        let matmul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matmul Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_sources::MATMUL.into()),
        });

        let spmv_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SpMV Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_sources::SPMV.into()),
        });

        let sparse_kernels_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sparse Kernels Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_sources::SPARSE_KERNELS.into()),
        });

        let conv2d_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Conv2D Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_sources::CONV2D.into()),
        });

        let element_wise_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Element Wise Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_sources::ELEMENT_WISE.into()),
        });

        let reduction_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduction Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_sources::REDUCTION.into()),
        });

        // Create compute pipelines
        let binary_ops_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Binary Ops Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Binary Ops Layout"),
                bind_group_layouts: &[&binary_ops_layout],
                push_constant_ranges: &[],
            })),
            module: &binary_ops_module,
            entry_point: "main",
        });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matmul Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Matmul Layout"),
                bind_group_layouts: &[&matmul_layout],
                push_constant_ranges: &[],
            })),
            module: &matmul_module,
            entry_point: "main",
        });

        let spmv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpMV Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SpMV Layout"),
                bind_group_layouts: &[&spmv_layout],
                push_constant_ranges: &[],
            })),
            module: &spmv_module,
            entry_point: "main",
        });

        let spmm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpMM Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SpMM Layout"),
                bind_group_layouts: &[&spmm_layout],
                push_constant_ranges: &[],
            })),
            module: &sparse_kernels_module,
            entry_point: "spmm_kernel",
        });

        let conv2d_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Conv2D Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Conv2D Layout"),
                bind_group_layouts: &[&conv2d_layout],
                push_constant_ranges: &[],
            })),
            module: &conv2d_module,
            entry_point: "main",
        });

        let element_wise_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Element Wise Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Element Wise Layout"),
                bind_group_layouts: &[&element_wise_layout],
                push_constant_ranges: &[],
            })),
            module: &element_wise_module,
            entry_point: "main",
        });

        let reduction_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reduction Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Reduction Layout"),
                bind_group_layouts: &[&reduction_layout],
                push_constant_ranges: &[],
            })),
            module: &reduction_module,
            entry_point: "main",
        });

        Ok(Some(Self {
            device,
            queue,
            binary_ops_pipeline,
            matmul_pipeline,
            spmv_pipeline,
            binary_ops_layout,
            matmul_layout,
            spmv_layout,
            spmm_pipeline,
            spmm_layout,
            conv2d_pipeline,
            conv2d_layout,
            element_wise_pipeline,
            element_wise_layout,
            reduction_pipeline,
            reduction_layout,
        }))
    }

    /// Create bind group layout for binary operations
    fn create_binary_ops_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Binary Ops Bind Group Layout"),
            entries: &[
                // lhs input
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
                // rhs input
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
                // output
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
                // op_type uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for matrix multiplication
    fn create_matmul_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matmul Bind Group Layout"),
            entries: &[
                // lhs matrix
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
                // rhs matrix
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
                // output matrix
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
                // dims uniform [M, K, N]
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for SpMV
    fn create_spmv_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpMV Bind Group Layout"),
            entries: &[
                // values
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
                // col_indices
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
                // row_ptrs
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
                // vector
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
                // output
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
                // uniforms
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
        })
    }

    /// Create bind group layout for SpMM
    fn create_spmm_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpMM Bind Group Layout"),
            entries: &[
                // csr_data (values)
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
                // csr_indices
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
                // csr_indptr
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
                // matrix_b (Dense)
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
                // matrix_c (Output)
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
                // matrix_info (Uniform)
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
        })
    }

    /// Create bind group layout for Conv2D
    fn create_conv2d_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Conv2D Bind Group Layout"),
            entries: &[
                // input
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // weight
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for Element-wise operations
    fn create_reduction_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reduction Bind Group Layout"),
            entries: &[
                // input
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
                // output (partial reductions)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for Element-wise operations
    fn create_element_wise_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Element Wise Bind Group Layout"),
            entries: &[
                // input
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
                // output
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // op_type uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Execute binary operation (add or multiply) on GPU
    ///
    /// # Arguments
    /// * `lhs` - Left operand array
    /// * `rhs` - Right operand array
    /// * `op` - Operation type: 0 = add, 1 = multiply
    ///
    /// # Returns
    /// Result array with same length as inputs
    pub fn binary_op(&self, lhs: &[f32], rhs: &[f32], op: u32) -> Result<Vec<f32>> {
        if lhs.len() != rhs.len() {
            return Err(crate::BackendError::InvalidInput(
                "Binary op inputs must have same length".to_string(),
            ));
        }

        let len = lhs.len();
        
        // Create GPU buffers
        let lhs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LHS Buffer"),
            contents: bytemuck::cast_slice(lhs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let rhs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RHS Buffer"),
            contents: bytemuck::cast_slice(rhs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let op_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Op Type Buffer"),
            contents: bytemuck::cast_slice(&[op]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Ops Bind Group"),
            layout: &self.binary_ops_layout,
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
                    resource: op_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute
        let workgroups = ((len as u32 + 255) / 256, 1, 1);
        self.execute_compute(&self.binary_ops_pipeline, &bind_group, workgroups)?;

        // Read back results
        self.read_buffer(&output_buffer, len)
    }

    /// Execute addition on GPU
    pub fn add(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
        self.binary_op(lhs, rhs, 0)
    }

    /// Execute multiplication on GPU
    pub fn mul(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
        self.binary_op(lhs, rhs, 1)
    }

    /// Execute element-wise unary operation on GPU
    pub fn unary_op(&self, input: &[f32], op: u32) -> Result<Vec<f32>> {
        let len = input.len();
        if len == 0 {
            return Ok(vec![]);
        }
        
        // Create GPU buffers
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let op_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Op Type Buffer"),
            contents: bytemuck::cast_slice(&[op]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Element Wise Bind Group"),
            layout: &self.element_wise_layout,
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
                    resource: op_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute
        let workgroups = ((len as u32 + 255) / 256, 1, 1);
        self.execute_compute(&self.element_wise_pipeline, &bind_group, workgroups)?;

        // Read back results
        self.read_buffer(&output_buffer, len)
    }



    /// Execute reduction on GPU (returns partial results, caller usually finishes on CPU for small size)
    pub fn reduce(&self, input: &[f32], op: u32, size: usize) -> Result<Vec<f32>> {
        if size == 0 {
            return Ok(vec![]);
        }
        
        // Workgroup size 256
        let workgroup_size = 256;
        let num_workgroups = (size + workgroup_size - 1) / workgroup_size;
        
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reduction Input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = num_workgroups * std::mem::size_of::<f32>();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reduction Output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniforms = [op, size as u32];
        let uniforms_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reduction Uniforms"),
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reduction Bind Group"),
            layout: &self.reduction_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uniforms_buffer.as_entire_binding() },
            ],
        });

        let workgroups = (num_workgroups as u32, 1, 1);
        self.execute_compute(&self.reduction_pipeline, &bind_group, workgroups)?;

        self.read_buffer(&output_buffer, num_workgroups)
    }

    /// Execute matrix multiplication on GPU
    ///
    /// Computes C = A × B where A is (M×K) and B is (K×N)
    pub fn matmul(&self, lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        // Create GPU buffers
        let lhs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LHS Matrix"),
            contents: bytemuck::cast_slice(lhs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let rhs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RHS Matrix"),
            contents: bytemuck::cast_slice(rhs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Matrix"),
            size: (m * n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dims = [m as u32, k as u32, n as u32];
        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dims Buffer"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
            layout: &self.matmul_layout,
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

        // Execute with 8x8 workgroups
        let workgroups = (
            ((m as u32) + 7) / 8,
            ((n as u32) + 7) / 8,
            1,
        );
        self.execute_compute(&self.matmul_pipeline, &bind_group, workgroups)?;

        // Read back results
        self.read_buffer(&output_buffer, m * n)
    }

    /// Execute sparse matrix-vector multiplication on GPU (CSR format)
    pub fn spmv_csr(
        &self,
        values: &[f32],
        col_indices: &[u32],
        row_ptrs: &[u32],
        vector: &[f32],
        num_rows: usize,
    ) -> Result<Vec<f32>> {
        // Create GPU buffers
        let values_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CSR Values"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let col_indices_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CSR Col Indices"),
            contents: bytemuck::cast_slice(col_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let row_ptrs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CSR Row Ptrs"),
            contents: bytemuck::cast_slice(row_ptrs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let vector_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Vector"),
            contents: bytemuck::cast_slice(vector),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Vector"),
            size: (num_rows * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniforms = [num_rows as u32];
        let uniforms_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpMV Uniforms"),
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SpMV Bind Group"),
            layout: &self.spmv_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: col_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_ptrs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vector_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniforms_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute
        let workgroups = (((num_rows as u32) + 255) / 256, 1, 1);
        self.execute_compute(&self.spmv_pipeline, &bind_group, workgroups)?;

        // Read back results
        self.read_buffer(&output_buffer, num_rows)
    }

    /// Execute compute pipeline
    fn execute_compute(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Execute sparse-dense matrix multiplication on GPU (SpMM)
    /// A (Sparse CSR) * B (Dense) = C (Dense)
    pub fn spmm_csr(
        &self,
        values: &[f32],
        col_indices: &[u32],
        row_ptrs: &[u32],
        matrix_b: &[f32],
        m: usize, // Rows of A (and C)
        n: usize, // Cols of B (and C)
    ) -> Result<Vec<f32>> {
        // Create GPU buffers
        let values_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CSR Values"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let col_indices_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CSR Col Indices"),
            contents: bytemuck::cast_slice(col_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let row_ptrs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CSR Row Ptrs"),
            contents: bytemuck::cast_slice(row_ptrs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let matrix_b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dense Matrix B"),
            contents: bytemuck::cast_slice(matrix_b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = m * n;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Matrix C"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Matrix Info: rows (m), cols (n), nnz (ignored)
        let matrix_info = [m as u32, n as u32, values.len() as u32, 0]; // Padding to 16 bytes alignment? u32 x 3 is 12 bytes. Uniforms need 16 byte alignment usually?
        // standard layout: struct CsrMatrix { rows: u32, cols: u32, nnz: u32 }
        // WGSL align of u32 is 4. Struct align is 4. Size 12.
        // BUT Uniform buffer bindings usually require 16-byte alignment of the *binding offset* (not size), but size should strictly encompass members.
        // It's safer to pad to 16 bytes (4 x u32).
        
        let matrix_info_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix Info"),
            contents: bytemuck::cast_slice(&matrix_info),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SpMM Bind Group"),
            layout: &self.spmm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: col_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_ptrs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: matrix_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: matrix_info_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute
        // Workgroup size (256, 1, 1). Each global_id.x is a row of A.
        // Dispatch m rows.
        let workgroups = (((m as u32) + 255) / 256, 1, 1);
        self.execute_compute(&self.spmm_pipeline, &bind_group, workgroups)?;

        // Read back results
        self.read_buffer(&output_buffer, output_size)
    }

    /// Execute 2D Convolution
    pub fn conv2d(
        &self,
        input: &[f32],
        weight: &[f32],
        input_dims: &[u32; 4], // N, C, H, W
        weight_dims: &[u32; 4], // OutC, InC, KH, KW
    ) -> Result<Vec<f32>> {
        let batch_size = input_dims[0];
        let in_channels = input_dims[1];
        let input_h = input_dims[2];
        let input_w = input_dims[3];
        
        let out_channels = weight_dims[0];
        // weight_dims[1] should match in_channels
        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];
        
        let output_h = input_h.saturating_sub(kernel_h) + 1;
        let output_w = input_w.saturating_sub(kernel_w) + 1;
        
        let output_size = (batch_size * out_channels * output_h * output_w) as usize;
        
        if output_size == 0 {
            return Ok(vec![]);
        }

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let weight_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weight Buffer"),
            contents: bytemuck::cast_slice(weight),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Pad uniforms to 16 bytes? struct has more than 4 u32s.
        // struct ConvUniforms {
        //    batch_size, in_channels, out_channels, input_h,
        //    input_w, kernel_h, kernel_w, output_h,
        //    output_w, ...
        // }
        // We pass 16 vals (some dummy)
        let uniforms: [u32; 16] = [
            batch_size, in_channels, out_channels, input_h,
            input_w, kernel_h, kernel_w, output_h,
            output_w, 1, 1, 0, 0, 1, 1, 1 // stride=1, pad=0, dil=1, groups=1
        ];
        
        let uniforms_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms Buffer"),
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Conv2D Bind Group"),
            layout: &self.conv2d_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uniforms_buffer.as_entire_binding() },
            ],
        });
        
        // Dispatch (W, H, B*C)
        let workgroups = (
            (output_w + 7) / 8,
            (output_h + 7) / 8,
            (batch_size * out_channels), 
        );
        
        self.execute_compute(&self.conv2d_pipeline, &bind_group, workgroups)?;
        
        self.read_buffer(&output_buffer, output_size)
    }

    /// Read buffer data back to CPU
    fn read_buffer(&self, buffer: &wgpu::Buffer, len: usize) -> Result<Vec<f32>> {
        // Create staging buffer for readback
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy to staging
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (len * std::mem::size_of::<f32>()) as u64);
        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let (tx, rx) = std::sync::mpsc::channel();
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        if rx.recv().unwrap().is_ok() {
            let result: Vec<f32> = {
                let data = slice.get_mapped_range();
                bytemuck::cast_slice(&data).to_vec()
            };
            staging.unmap();
            Ok(result)
        } else {
            Err(crate::BackendError::GpuError("Failed to read buffer".to_string()))
        }
    }
}

/// Global GPU executor instance (lazy initialized)
static GPU_EXECUTOR: std::sync::OnceLock<Option<GpuDenseExecutor>> = std::sync::OnceLock::new();

/// Get the global GPU executor, initializing if needed
pub fn get_gpu_executor() -> Option<&'static GpuDenseExecutor> {
    // This is blocking but only happens once
    GPU_EXECUTOR.get_or_init(|| {
        // Use pollster to block on async initialization
        pollster::block_on(async {
            GpuDenseExecutor::new().await.ok().flatten()
        })
    }).as_ref()
}
