//! GPU backend infrastructure (CPU FALLBACK ONLY - NO GPU ACCELERATION)

use super::{Backend, BackendData, BackendError, Device, Result, Tensor, TensorData};
use coeus_dtype::Dtype;
use std::sync::Arc;

/// GPU backend infrastructure - CURRENTLY CPU FALLBACK ONLY
///
/// # WARNING: No GPU Acceleration Implemented
/// This backend provides wgpu infrastructure but performs all operations on CPU.
/// GPU acceleration is planned for future implementation but currently unavailable.
///
/// All operations transfer data to CPU, perform computation, then transfer back.
/// This provides no performance benefit and increased latency.
#[derive(Debug)]
pub struct GpuBackend {
    /// WGPU instance (reserved for future GPU operations)
    _instance: wgpu::Instance,
    /// Physical adapter
    adapter: wgpu::Adapter,
    /// Logical device
    device: wgpu::Device,
    /// Command queue
    queue: wgpu::Queue,
}

impl GpuBackend {
    /// Create a new GPU backend
    ///
    /// This initializes the wgpu instance, adapter, and device.
    /// Returns an error if no suitable GPU is found.
    pub async fn new() -> Result<Self> {
        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| BackendError::gpu_error("No suitable GPU adapter found"))?;

        // Request device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: Some("Coeus GPU Backend"),
                },
                None,
            )
            .await
            .map_err(|e| BackendError::gpu_error(format!("Failed to create device: {e:?}")))?;

        Ok(Self {
            _instance: instance,
            adapter,
            device,
            queue,
        })
    }

    /// Get adapter information for debugging
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }
}

#[async_trait::async_trait]
impl<T: Dtype + bytemuck::Pod> Backend<T> for GpuBackend {
    fn device(&self) -> Device {
        Device::Gpu
    }

    async fn allocate(&self, shape: &[usize]) -> Result<Arc<TensorData<T>>> {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * std::mem::size_of::<T>();

        // For now, create a placeholder buffer - full GPU implementation would require
        // proper buffer creation with wgpu
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor Buffer"),
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Arc::new(TensorData {
            shape: shape.to_vec(),
            data: BackendData::Gpu(buffer),
        }))
    }

    async fn copy_from_host(&self, data: &[T], shape: &[usize]) -> Result<Tensor<T>> {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * std::mem::size_of::<T>();

        // Create staging buffer for host-to-device transfer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Host-to-Device Staging Buffer"),
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });

        // Write data to staging buffer using safe bytemuck casting
        {
            let mut buffer_view = staging_buffer.slice(..).get_mapped_range_mut();
            let data_bytes: &[u8] = bytemuck::cast_slice(data);
            buffer_view.copy_from_slice(data_bytes);
        }
        staging_buffer.unmap();

        // Create device buffer
        let device_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Device Buffer"),
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Copy from staging to device buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(&staging_buffer, 0, &device_buffer, 0, size_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        Ok(Tensor {
            data: Arc::new(TensorData {
                shape: shape.to_vec(),
                data: BackendData::Gpu(device_buffer),
            }),
            shape: shape.to_vec(),
        })
    }

    async fn copy_to_host(&self, tensor: &Tensor<T>) -> Result<Vec<T>> {
        let BackendData::Gpu(buffer) = &tensor.data.data else {
            return Err(BackendError::DeviceMismatch {
                required: Device::Gpu,
                actual: Device::Cpu,
            });
        };

        let numel = tensor.numel();
        let size_bytes = numel * std::mem::size_of::<T>();

        // Create staging buffer for device-to-host transfer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Device-to-Host Staging Buffer"),
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy from device to staging buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        // Read data from staging buffer
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.receive().await.unwrap().unwrap();

        let data = {
            let buffer_view = buffer_slice.get_mapped_range();
            let bytes: &[u8] = bytemuck::cast_slice(&buffer_view);
            // Use safe bytemuck casting to convert back to typed slice
            bytemuck::cast_slice::<u8, T>(bytes).to_vec()
        };

        staging_buffer.unmap();
        Ok(data)
    }

    async fn add(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        // WARNING: CPU fallback - no GPU acceleration implemented
        // Transfer data to CPU for computation, then back to GPU
        let a_data = self.copy_to_host(a).await?;
        let b_data = self.copy_to_host(b).await?;

        let cpu_backend = crate::cpu::CpuBackend::new();
        let a_cpu = cpu_backend.copy_from_host(&a_data, a.shape()).await?;
        let b_cpu = cpu_backend.copy_from_host(&b_data, b.shape()).await?;
        let result_cpu = cpu_backend.add(&a_cpu, &b_cpu).await?;

        // Transfer result back to GPU
        let result_data = cpu_backend.copy_to_host(&result_cpu).await?;
        self.copy_from_host(&result_data, result_cpu.shape()).await
    }

    async fn sub(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        // Transfer data to CPU for computation, then back to GPU
        let a_data = self.copy_to_host(a).await?;
        let b_data = self.copy_to_host(b).await?;

        let cpu_backend = crate::cpu::CpuBackend::new();
        let a_cpu = cpu_backend.copy_from_host(&a_data, a.shape()).await?;
        let b_cpu = cpu_backend.copy_from_host(&b_data, b.shape()).await?;
        let result_cpu = cpu_backend.sub(&a_cpu, &b_cpu).await?;

        // Transfer result back to GPU
        let result_data = cpu_backend.copy_to_host(&result_cpu).await?;
        self.copy_from_host(&result_data, result_cpu.shape()).await
    }

    async fn mul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        // Transfer data to CPU for computation, then back to GPU
        let a_data = self.copy_to_host(a).await?;
        let b_data = self.copy_to_host(b).await?;

        let cpu_backend = crate::cpu::CpuBackend::new();
        let a_cpu = cpu_backend.copy_from_host(&a_data, a.shape()).await?;
        let b_cpu = cpu_backend.copy_from_host(&b_data, b.shape()).await?;
        let result_cpu = cpu_backend.mul(&a_cpu, &b_cpu).await?;

        // Transfer result back to GPU
        let result_data = cpu_backend.copy_to_host(&result_cpu).await?;
        self.copy_from_host(&result_data, result_cpu.shape()).await
    }

    async fn div(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        // Transfer data to CPU for computation, then back to GPU
        let a_data = self.copy_to_host(a).await?;
        let b_data = self.copy_to_host(b).await?;

        let cpu_backend = crate::cpu::CpuBackend::new();
        let a_cpu = cpu_backend.copy_from_host(&a_data, a.shape()).await?;
        let b_cpu = cpu_backend.copy_from_host(&b_data, b.shape()).await?;
        let result_cpu = cpu_backend.div(&a_cpu, &b_cpu).await?;

        // Transfer result back to GPU
        let result_data = cpu_backend.copy_to_host(&result_cpu).await?;
        self.copy_from_host(&result_data, result_cpu.shape()).await
    }

    async fn matmul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        // WARNING: CPU fallback - no GPU acceleration implemented
        // Validate matrix multiplication shapes
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(BackendError::invalid_operation(
                "Matrix multiplication requires at least 2D tensors",
            ));
        }

        let _m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let _n = b_shape[b_shape.len() - 1];

        if k != b_shape[b_shape.len() - 2] {
            return Err(BackendError::invalid_operation(format!(
                "Incompatible shapes for matrix multiplication: {:?} @ {:?}",
                a_shape, b_shape
            )));
        }

        // Transfer data to CPU for computation, then back to GPU
        let a_data = self.copy_to_host(a).await?;
        let b_data = self.copy_to_host(b).await?;

        let cpu_backend = crate::cpu::CpuBackend::new();
        let a_cpu = cpu_backend.copy_from_host(&a_data, a.shape()).await?;
        let b_cpu = cpu_backend.copy_from_host(&b_data, b.shape()).await?;
        let result_cpu = cpu_backend.matmul(&a_cpu, &b_cpu).await?;

        // Transfer result back to GPU
        let result_data = cpu_backend.copy_to_host(&result_cpu).await?;
        self.copy_from_host(&result_data, result_cpu.shape()).await
    }

    async fn transpose(&self, tensor: &Tensor<T>, dim0: usize, dim1: usize) -> Result<Tensor<T>> {
        // CPU fallback for transpose
        let cpu_backend = crate::cpu::CpuBackend::new();
        cpu_backend.transpose(tensor, dim0, dim1).await
    }

    async fn sum_dim(&self, tensor: &Tensor<T>, dim: usize) -> Result<Tensor<T>> {
        // CPU fallback for sum_dim
        let cpu_backend = crate::cpu::CpuBackend::new();
        cpu_backend.sum_dim(tensor, dim).await
    }

    async fn mean_dim(&self, tensor: &Tensor<T>, dim: usize) -> Result<Tensor<T>> {
        // CPU fallback for mean_dim
        let cpu_backend = crate::cpu::CpuBackend::new();
        cpu_backend.mean_dim(tensor, dim).await
    }

    async fn cat(&self, _tensors: &[&Tensor<T>], _dim: usize) -> Result<Tensor<T>> {
        Err(BackendError::invalid_operation(
            "GPU cat not yet implemented",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[tokio::test]
    async fn test_gpu_backend_creation() {
        // Skip test if no GPU available
        let backend = GpuBackend::new().await;
        match backend {
            Ok(gpu_backend) => {
                assert_eq!(
                    <GpuBackend as Backend<f32>>::device(&gpu_backend),
                    Device::Gpu
                );
                println!(
                    "GPU Backend created with adapter: {:?}",
                    gpu_backend.adapter_info()
                );
            }
            Err(_) => {
                println!("GPU backend not available, skipping test");
            }
        }
    }

    #[tokio::test]
    async fn test_gpu_tensor_allocation() {
        let backend = match GpuBackend::new().await {
            Ok(b) => b,
            Err(_) => {
                println!("GPU not available, skipping test");
                return;
            }
        };

        let tensor: Tensor<f32> = backend.zeros(&[2, 3]).await.unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
    }

    #[tokio::test]
    async fn test_gpu_memory_transfer() {
        let backend = match GpuBackend::new().await {
            Ok(b) => b,
            Err(_) => {
                println!("GPU not available, skipping test");
                return;
            }
        };

        // Test data
        let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2, 3];

        // Copy to GPU
        let gpu_tensor = backend
            .copy_from_host(&original_data, &shape)
            .await
            .unwrap();

        // Copy back to CPU
        let cpu_data = backend.copy_to_host(&gpu_tensor).await.unwrap();

        // Verify data integrity
        assert_eq!(cpu_data.len(), original_data.len());
        for (original, copied) in original_data.iter().zip(cpu_data.iter()) {
            assert_relative_eq!(*original, *copied, epsilon = 1e-6);
        }
    }

    #[tokio::test]
    async fn test_gpu_buffer_creation() {
        let backend = match GpuBackend::new().await {
            Ok(b) => b,
            Err(_) => {
                println!("GPU not available, skipping test");
                return;
            }
        };

        // Create a GPU buffer
        let tensor_data: Arc<TensorData<f32>> = backend.allocate(&[100]).await.unwrap();
        assert_eq!(tensor_data.shape, vec![100]);

        // Verify it's a GPU buffer
        match tensor_data.data {
            BackendData::Gpu(_) => {
                // GPU buffer created successfully
            }
            _ => panic!("Expected GPU buffer"),
        }
    }

    #[tokio::test]
    async fn test_gpu_tensor_operations_with_cpu_fallback() {
        let backend = match GpuBackend::new().await {
            Ok(b) => b,
            Err(_) => {
                println!("GPU not available, skipping test");
                return;
            }
        };

        // Create test tensors
        let a: Tensor<f32> = backend.zeros(&[2, 2]).await.unwrap();
        let b: Tensor<f32> = backend.zeros(&[2, 2]).await.unwrap();

        // Test that operations now work with CPU fallbacks
        let add_result = backend.add(&a, &b).await;
        assert!(
            add_result.is_ok(),
            "GPU addition should work with CPU fallback"
        );

        let matmul_result = backend.matmul(&a, &b).await;
        assert!(
            matmul_result.is_ok(),
            "GPU matrix multiplication should work with CPU fallback"
        );
    }
}
