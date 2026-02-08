pub use crate::core::DeviceInfo;
use core::fmt;

#[cfg(feature = "std")]
use std::string::String;

/// Device enumeration for backend identification
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Device {
    /// CPU device
    Cpu,
    /// GPU device with detailed information
    Gpu {
        /// GPU device name
        name: String,
        /// GPU vendor ID
        vendor: u32,
        /// GPU device ID
        device: u32,
        /// Backend API (Vulkan, Metal, DX12, etc.)
        backend: &'static str,
    },
    /// Neural processing unit
    Npu {
        /// NPU device name
        name: String,
        /// Manufacturer
        manufacturer: String,
        /// Number of compute units
        compute_units: usize,
        /// Peak performance in TOPS
        peak_tops: f32,
        /// On-chip memory in MB
        memory_mb: usize,
    },
    /// Tensor processing unit
    Tpu {
        /// TPU device name
        name: String,
        /// TPU generation/version
        generation: String,
        /// Number of cores
        cores: usize,
        /// Peak performance in TOPS
        peak_tops: f32,
        /// Memory capacity in GB
        memory_gb: usize,
    },
}

impl Device {
    /// Returns the device name
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu { name, .. } => name,
            Self::Npu { name, .. } => name,
            Self::Tpu { name, .. } => name,
        }
    }

    /// Returns memory capacity in GB
    pub fn memory_gb(&self) -> usize {
        match self {
            Device::Cpu => 16,       // Assume 16GB system RAM for CPU
            Device::Gpu { .. } => 8, // Assume 8GB VRAM for GPU
            Device::Npu { memory_mb, .. } => memory_mb / 1024,
            Device::Tpu { memory_gb, .. } => *memory_gb,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl DeviceInfo for Device {
    fn name(&self) -> &str {
        self.name()
    }

    fn memory_total(&self) -> Option<usize> {
        Some(self.memory_gb() * 1024 * 1024 * 1024)
    }

    fn memory_available(&self) -> Option<usize> {
        // For now assume all memory available or unknown
        None
    }

    fn compute_capability(&self) -> Option<String> {
        match self {
            Device::Gpu { backend, .. } => Some(backend.to_string()),
            Device::Tpu { generation, .. } => Some(generation.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    use std::string::ToString;

    #[test]
    fn test_device_name() {
        assert_eq!(Device::Cpu.name(), "cpu");
        let gpu_device = Device::Gpu {
            name: "NVIDIA RTX 3080".to_string(),
            vendor: 4318,
            device: 8225,
            backend: "Vulkan",
        };
        assert_eq!(gpu_device.name(), "NVIDIA RTX 3080");

        let npu_device = Device::Npu {
            name: "Apple Neural Engine".to_string(),
            manufacturer: "Apple".to_string(),
            compute_units: 16,
            peak_tops: 15.8,
            memory_mb: 8,
        };
        assert_eq!(npu_device.name(), "Apple Neural Engine");

        let tpu_device = Device::Tpu {
            name: "Cloud TPU v4".to_string(),
            generation: "v4".to_string(),
            cores: 4,
            peak_tops: 275.0,
            memory_gb: 32,
        };
        assert_eq!(tpu_device.name(), "Cloud TPU v4");
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_device_display() {
        use std::format;
        let cpu = Device::Cpu;
        assert_eq!(format!("{cpu}"), "cpu");
    }
}
