//! Metal backend stub (forward to wgpu for now).

use super::Backend;
use wgpu::Device; // Assume wgpu-metal integration

pub struct MetalBackend {
    wgpu_device: Device,
}

impl MetalBackend {
    pub fn new(wgpu_device: Device) -> Self {
        Self { wgpu_device }
    }
}

impl Backend for MetalBackend {
    type Dtype = f32;
    type TensorData = Vec<Self::Dtype>; // Stub: forward to wgpu

    fn create_tensor_data(&self, data: &[Self::Dtype], shape: &[usize]) -> Self::TensorData {
        // Forward to wgpu buffer via adapter
        todo!("Full Metal impl; stub forwards to wgpu")
    }

    fn add(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData {
        // Delegate to wgpu GPU backend
        let wgpu_backend = GpuBackend::new(self.wgpu_device.clone(), /* queue */);
        wgpu_backend.add(a, b)
    }

    // ...forward other ops...
}
