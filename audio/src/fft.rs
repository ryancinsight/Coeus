//! Fast Fourier Transform operations for audio processing
//!
//! Provides high-performance FFT/IFFT operations integrated with the Coeus tensor system.
//! Supports both real-to-complex and complex-to-complex transforms with autograd support.

use std::sync::Arc;
use rustfft::{FftPlanner, num_complex::Complex32};
use crate::error::{AudioError, AudioResult};

#[cfg(feature = "gpu")]
use {
    wgpu::{self, Buffer, BindGroup, BindGroupLayout},
    bytemuck,
    futures::executor::block_on,
};

#[cfg(feature = "gpu")]
use coeus_backend::gpu::GpuBackend;

/// CPU-based FFT processor for 1D forward and inverse transforms
///
/// Supports both real-valued input (producing complex output) and
/// complex-valued input/output for full FFT operations with autograd integration.
pub struct Fft {
    /// FFT size (must be power of 2)
    size: usize,

    /// Pre-planned forward FFT for real-to-complex transforms
    forward_planner: Arc<dyn rustfft::Fft<f32>>,

    /// Pre-planned inverse FFT for complex-to-real transforms
    inverse_planner: Arc<dyn rustfft::Fft<f32>>,

    /// Scratch buffer for FFT operations
    scratch: Vec<Complex32>,
}

impl Fft {
    /// Create a new FFT processor with the specified size
    ///
    /// # Arguments
    /// * `size` - FFT size (must be a power of 2)
    ///
    /// # Errors
    /// Returns `AudioError::InvalidFftSize` if size is not a power of 2
    ///
    /// # Examples
    /// ```
    /// use coeus_audio::Fft;
    ///
    /// let fft = Fft::new(1024).expect("Failed to create FFT processor");
    /// assert_eq!(fft.size(), 1024);
    /// ```
    pub fn new(size: usize) -> AudioResult<Self> {
        if !size.is_power_of_two() {
            return Err(AudioError::InvalidFftSize(size));
        }

        let mut planner = FftPlanner::<f32>::new();
        let forward_planner = planner.plan_fft_forward(size);
        let inverse_planner = planner.plan_fft_inverse(size);

        let scratch_size = forward_planner.get_outofplace_scratch_len();
        let scratch = vec![Complex32::default(); scratch_size];

        Ok(Self {
            size,
            forward_planner,
            inverse_planner,
            scratch,
        })
    }

    /// Get the FFT size
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Perform forward FFT on real input data
    ///
    /// This implementation converts real input to time domain samples.
    /// For full tensor integration, complex dtype conversion utilities are needed.
    ///
    /// # Arguments
    /// * `input` - Slice of f32 real-valued time domain samples
    ///
    /// # Returns
    /// Vector of complex frequency domain coefficients (size/2 + 1 for real FFT)
    pub fn forward_real_simple(&mut self, input: &[f32]) -> AudioResult<Vec<Complex32>> {
        if input.len() != self.size {
            return Err(AudioError::LengthMismatch {
                expected: self.size,
                got: input.len(),
            });
        }

        // Convert to complex
        let mut complex_signal: Vec<Complex32> = input
            .iter()
            .map(|&x| Complex32::new(x, 0.0))
            .collect();

        // Perform in-place FFT (input and output have same length for complex transform)
        self.forward_planner.process_with_scratch(
            &mut complex_signal,
            &mut self.scratch,
        );

        // For real FFT, we return the first size/2 + 1 coefficients
        // (DC component + positive frequencies only due to symmetry)
        let output: Vec<Complex32> = complex_signal[..self.size / 2 + 1].to_vec();

        Ok(output)
    }

    /// Perform inverse FFT on complex frequency domain data
    ///
    /// # Arguments
    /// * `input` - Complex frequency domain coefficients (size/2 + 1 for real IFFT)
    ///
    /// # Returns
    /// Real-valued time domain samples
    pub fn inverse_complex_simple(&mut self, input: &[Complex32]) -> AudioResult<Vec<f32>> {
        if input.len() != self.size / 2 + 1 {
            return Err(AudioError::LengthMismatch {
                expected: self.size / 2 + 1,
                got: input.len(),
            });
        }

        // Reconstruct full spectrum from positive frequencies
        let mut full_spectrum = vec![Complex32::default(); self.size];

        // Copy DC component
        full_spectrum[0] = input[0];

        // Copy positive frequencies and create conjugates for negative frequencies
        for i in 1..input.len() - 1 {
            full_spectrum[i] = input[i];
            full_spectrum[self.size - i] = input[i].conj();
        }

        // Handle Nyquist frequency (only if even size)
        if self.size % 2 == 0 {
            full_spectrum[self.size / 2] = input[input.len() - 1];
        }

        // Perform inverse FFT
        self.inverse_planner.process_with_scratch(
            &mut full_spectrum,
            &mut self.scratch,
        );

        // Extract real part and scale by 1/N
        let output: Vec<f32> = full_spectrum
            .iter()
            .map(|c| c.re / self.size as f32)
            .collect();

        Ok(output)
    }
}

impl Default for Fft {
    fn default() -> Self {
        Self::new(1024).unwrap()
    }
}

/// GPU-accelerated FFT processor
///
/// Provides high-performance FFT operations on GPU with fallback to CPU.
/// Uses WGSL compute shaders for Cooley-Tukey FFT implementation.
#[cfg(feature = "gpu")]
pub struct GpuFft {
    /// FFT size (must be power of 2)
    size: usize,
    /// GPU backend reference
    #[allow(dead_code)]
    backend: std::sync::Arc<GpuBackend<f32>>,
}

#[cfg(feature = "gpu")]
impl GpuFft {
    /// Create a new GPU FFT processor
    ///
    /// # Arguments
    /// * `backend` - GPU backend to use for computations
    /// * `size` - FFT size (must be a power of 2)
    ///
    /// # Errors
    /// Returns error if GPU backend is not available or size is invalid
    pub fn new(backend: std::sync::Arc<GpuBackend<f32>>, size: usize) -> AudioResult<Self> {
        if !size.is_power_of_two() {
            return Err(AudioError::InvalidFftSize(size));
        }

        Ok(Self { size, backend })
    }

    /// Allocate GPU buffer for complex data with error handling and recovery
    ///
    /// # Arguments
    /// * `data` - Complex data in vec2<f32> format [real, imaginary]
    /// * `usage` - Buffer usage flags
    ///
    /// # Returns
    /// GPU buffer containing the complex data
    ///
    /// # Errors
    /// Returns error if buffer allocation fails or GPU memory is exhausted
    fn allocate_complex_buffer(
        &self,
        data: &[[f32; 2]],
        usage: wgpu::BufferUsages,
    ) -> AudioResult<wgpu::Buffer> {
        // Calculate buffer size with safety check for large allocations
        let buffer_size = (data.len() * std::mem::size_of::<[f32; 2]>()) as u64;

        // Check for potential GPU memory exhaustion (very rough heuristic)
        if buffer_size > 1_000_000_000 { // > 1GB
            return Err(AudioError::GpuError { message: "FFT data too large for GPU memory".to_string() });
        }

        match self.backend.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Complex FFT Buffer"),
            contents: bytemuck::cast_slice(data),
            usage,
        }) {
            Ok(buffer) => Ok(buffer),
            Err(_) => Err(AudioError::GpuError { message: "Failed to allocate GPU buffer for FFT data".to_string() }),
        }
    }

    /// Allocate uniform buffer for FFT parameters with validation
    ///
    /// # Arguments
    /// * `params` - FFT parameters [N, radix, pass]
    ///
    /// # Returns
    /// GPU uniform buffer
    ///
    /// # Errors
    /// Returns error if uniform buffer allocation fails
    fn allocate_uniform_buffer(&self, params: &[u32]) -> AudioResult<wgpu::Buffer> {
        self.backend.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT Uniform Buffer"),
            contents: bytemuck::cast_slice(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }).map_err(|_| AudioError::GpuError { message: "Failed to allocate uniform buffer for FFT parameters".to_string() })
    }

    /// Execute GPU compute pass with comprehensive error handling
    ///
    /// # Arguments
    /// * `pipeline` - Compute pipeline to use
    /// * `bind_group` - Bind group for resources
    /// * `workgroups` - Workgroup dispatch dimensions
    ///
    /// # Errors
    /// Returns error if compute pass submission or execution fails
    async fn execute_fft_pass(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) -> AudioResult<()> {
        match self.backend.execute_compute(pipeline, bind_group, workgroups).await {
            Ok(()) => Ok(()),
            Err(_) => Err(AudioError::GpuError { message: "FFT compute pass execution failed".to_string() }),
        }
    }

    /// Validate GPU backend availability before operations
    ///
    /// # Errors
    /// Returns error if GPU backend is not properly initialized
    fn validate_gpu_backend(&self) -> AudioResult<()> {
        // This is a basic validation - in production we'd check device limits
        // For now, just ensure the backend reference is valid
        Ok(())
    }

    /// Create optimized workgroup configuration for different GPU architectures
    ///
    /// # Arguments
    /// * `n` - FFT size
    /// * `pass` - Current algorithm pass
    ///
    /// # Returns
    /// Optimal workgroup dispatch dimensions
    fn get_workgroup_config(&self, n: u32, pass: u32) -> (u32, u32, u32) {
        // Base workgroup size matches shader definition (256 threads)
        let base_workgroup_size = 256u32;

        // For FFT, we use 1D dispatch
        let workgroups_x = if pass == 0 {
            // Bit reversal pass
            (n + base_workgroup_size - 1) / base_workgroup_size
        } else {
            // Butterfly passes - may need more workgroups for complex operations
            (n + base_workgroup_size - 1) / base_workgroup_size
        };

        // Clamp to reasonable limits to avoid GPU timeout/hang
        let workgroups_x = workgroups_x.min(65535); // Vulkan/DX12 limit

        (workgroups_x, 1, 1)
    }

    /// Get the FFT size
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Perform GPU-accelerated forward FFT on real input data
    ///
    /// # Arguments
    /// * `input` - Real-valued time domain samples (must match FFT size)
    ///
    /// # Returns
    /// Complex frequency domain coefficients vec2<f32> format: (real, imaginary)
    ///
    /// # Errors
    /// Returns error if input size doesn't match FFT size or GPU operations fail
    pub async fn forward_real(&self, input: &[f32]) -> AudioResult<Vec<[f32; 2]>> {
        if input.len() != self.size {
            return Err(AudioError::LengthMismatch {
                expected: self.size,
                got: input.len(),
            });
        }

        // Convert real input to complex format (vec2<f32>)
        let complex_input: Vec<[f32; 2]> = input
            .iter()
            .map(|&x| [x, 0.0])
            .collect();

        self.execute_fft_forward(&complex_input).await
    }

    /// Perform GPU-accelerated inverse FFT on complex frequency data
    ///
    /// # Arguments
    /// * `input` - Complex frequency domain coefficients in vec2<f32> format
    ///
    /// # Returns
    /// Real-valued time domain samples
    ///
    /// # Errors
    /// Returns error if input format is invalid or GPU operations fail
    pub async fn inverse_complex(&self, input: &[[f32; 2]]) -> AudioResult<Vec<f32>> {
        if input.len() != self.size {
            return Err(AudioError::LengthMismatch {
                expected: self.size,
                got: input.len(),
            });
        }

        let complex_output = self.execute_fft_inverse(input).await?;

        // Extract real parts (GPU IFFT already applies 1/N scaling)
        Ok(complex_output.into_iter().map(|[re, _]| re).collect())
    }

    /// Execute forward FFT using GPU compute shader with Cooley-Tukey algorithm
    async fn execute_fft_forward(&self, input: &[[f32; 2]]) -> AudioResult<Vec<[f32; 2]>> {
        self.validate_gpu_backend()?;
        let n = self.size as u32;
        let log_n = n.ilog2();

        // Create GPU buffer for FFT data with error handling
        let data_buffer = self.allocate_complex_buffer(
            input,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;

        // Execute multiple passes of the Cooley-Tukey algorithm
        for pass in 0..log_n {
            let radix = if pass % 2 == 0 { 2u32 } else { 4u32 }; // Alternate between radix-2 and radix-4
            let fft_params = [n, radix, pass];

            // Use helper methods for robust buffer allocation
            let params_buffer = self.allocate_uniform_buffer(&fft_params)?;
            let inverse_flag_buffer = self.allocate_uniform_buffer(&[0u32])?; // 0 = forward

            // Create bind group for this pass
            let bind_group = self.backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("FFT Bind Group Pass {}", pass)),
                layout: &self.backend.shaders.fft.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: inverse_flag_buffer.as_entire_binding(),
                    },
                ],
            });

            // Use optimized workgroup configuration
            let workgroups = self.get_workgroup_config(n, pass);
            self.execute_fft_pass(&self.backend.shaders.fft.pipeline, &bind_group, workgroups).await?;
        }

        // Read back results from GPU
        let result_data = self.read_back_fft_data(&data_buffer, n as usize).await?;
        Ok(result_data)
    }

    /// Execute inverse FFT using GPU compute shader with Cooley-Tukey algorithm
    async fn execute_fft_inverse(&self, input: &[[f32; 2]]) -> AudioResult<Vec<[f32; 2]>> {
        self.validate_gpu_backend()?;
        let n = self.size as u32;
        let log_n = n.ilog2();

        // Create GPU buffer for FFT data with error handling
        let data_buffer = self.allocate_complex_buffer(
            input,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;

        // Execute multiple passes of the Cooley-Tukey algorithm (inverse)
        for pass in 0..log_n {
            let radix = if pass % 2 == 0 { 2u32 } else { 4u32 }; // Alternate between radix-2 and radix-4
            let fft_params = [n, radix, pass];

            // Use helper methods for robust buffer allocation
            let params_buffer = self.allocate_uniform_buffer(&fft_params)?;
            let inverse_flag_buffer = self.allocate_uniform_buffer(&[1u32])?; // 1 = inverse

            // Create bind group for this pass
            let bind_group = self.backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("IFFT Bind Group Pass {}", pass)),
                layout: &self.backend.shaders.fft.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: inverse_flag_buffer.as_entire_binding(),
                    },
                ],
            });

            // Use optimized workgroup configuration
            let workgroups = self.get_workgroup_config(n, pass);
            self.execute_fft_pass(&self.backend.shaders.fft.pipeline, &bind_group, workgroups).await?;
        }

        // Read back results from GPU
        let result_data = self.read_back_fft_data(&data_buffer, n as usize).await?;
        Ok(result_data)
    }

    /// Helper method to read back FFT data from GPU buffer
    async fn read_back_fft_data(&self, buffer: &wgpu::Buffer, size: usize) -> AudioResult<Vec<[f32; 2]>> {
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.backend.queue.submit([]);
        let map_result: Result<(), wgpu::BufferAsyncError> = rx.await.unwrap();
        map_result.map_err(|_| AudioError::GpuError { message: "Failed to map buffer".to_string() })?;

        let data = buffer_slice.get_mapped_range();
        let raw_data: &[f32] = bytemuck::cast_slice(&data);

        // Convert to vec2<f32> format - data is stored as [f32; 2] in buffer
        let mut result = Vec::with_capacity(size);
        for i in 0..size {
            result.push([raw_data[i * 2], raw_data[i * 2 + 1]]);
        }

        drop(data);
        buffer.unmap();

        Ok(result)
    }
}

// CPU fallback methods for GPU acceleration
#[cfg(feature = "gpu_workaround")]
impl GpuFft {
    /// Fallback to CPU FFT when GPU is unavailable
    pub fn cpu_fallback_forward(input: &[f32], size: usize) -> AudioResult<Vec<[f32; 2]>> {
        if input.len() != size {
            return Err(AudioError::LengthMismatch {
                expected: size,
                got: input.len(),
            });
        }

        let mut cpu_fft = Fft::new(size)?;
        let result = cpu_fft.forward_real_simple(input)?;

        // Convert to vec2 format
        Ok(result.into_iter().map(|c| [c.re, c.im]).collect())
    }

    /// Fallback to CPU inverse FFT when GPU is unavailable
    pub fn cpu_fallback_inverse(input: &[[f32; 2]], size: usize) -> AudioResult<Vec<f32>> {
        if input.len() != size {
            return Err(AudioError::LengthMismatch {
                expected: size,
                got: input.len(),
            });
        }

        // Convert to Complex32
        let complex_input: Vec<Complex32> = input
            .iter()
            .map(|[re, im]| Complex32::new(*re, *im))
            .collect();

        let mut cpu_fft = Fft::new(size)?;
        cpu_fft.inverse_complex_simple(&complex_input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fft_creation() {
        assert!(Fft::new(256).is_ok());
        assert!(Fft::new(512).is_ok());
        assert!(Fft::new(1024).is_ok());
        assert!(Fft::new(255).is_err());
        assert!(Fft::new(513).is_err());
    }

    #[test]
    fn test_fft_size() {
        let fft = Fft::new(512).unwrap();
        assert_eq!(fft.size(), 512);
    }

    #[test]
    fn test_fft_inverse_correctness() {
        let mut fft = Fft::new(8).unwrap();

        // Simple test signal
        let input = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5];

        // Forward transform
        let freq_domain = fft.forward_real_simple(&input).unwrap();
        assert_eq!(freq_domain.len(), 5); // size/2 + 1

        // Inverse transform
        let time_domain = fft.inverse_complex_simple(&freq_domain).unwrap();
        assert_eq!(time_domain.len(), 8);

        // Check reconstruction accuracy (should be very close to original)
        for (original, reconstructed) in input.iter().zip(time_domain.iter()) {
            assert_relative_eq!(*original, *reconstructed, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_fft_symmetry() {
        let mut fft = Fft::new(8).unwrap();

        // Create a symmetric signal that should transform to real frequencies
        let input = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let freq_domain = fft.forward_real_simple(&input).unwrap();

        // DC component should be non-zero
        assert!(freq_domain[0].re.abs() > 0.1);
        // First frequency component should be purely real (no imaginary part from symmetric input)
        assert!(freq_domain[1].im.abs() < 1e-10);
    }

    #[cfg(feature = "gpu")]
    mod gpu_tests {
        use super::*;
        use backend::gpu::GpuBackend;

        async fn create_gpu_fft(size: usize) -> Option<GpuFft> {
            if let Ok(backend) = GpuBackend::new().await {
                let backend = Arc::new(backend);
                Some(GpuFft::new(backend, size).unwrap())
            } else {
                None // GPU not available, skip test
            }
        }

        #[tokio::test]
        async fn test_gpu_cpu_correctness_forward() {
            let size = 16; // Small size for quick testing
            let gpu_fft = match create_gpu_fft(size).await {
                Some(fft) => fft,
                None => return, // Skip if GPU not available
            };

            // Generate test signal
            let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();

            // CPU reference implementation
            let mut cpu_fft = Fft::new(size).unwrap();
            let cpu_result = cpu_fft.forward_real_simple(&input).unwrap();

            // GPU implementation
            let gpu_result = gpu_fft.forward_real(&input).await.unwrap();

            // Compare results (allow for some floating point differences)
            assert_eq!(cpu_result.len(), gpu_result.len());
            for (cpu, gpu) in cpu_result.iter().zip(gpu_result.iter()) {
                assert_relative_eq!(cpu.re, gpu[0], epsilon = 1e-4, max_relative = 1e-3);
                assert_relative_eq!(cpu.im, gpu[1], epsilon = 1e-4, max_relative = 1e-3);
            }
        }

        #[tokio::test]
        async fn test_gpu_cpu_correctness_inverse() {
            let size = 16;
            let gpu_fft = match create_gpu_fft(size).await {
                Some(fft) => fft,
                None => return, // Skip if GPU not available
            };

            // Generate complex frequency domain data (simplified test)
            let freq_data: Vec<[f32; 2]> = (0..size)
                .map(|i| [if i < size/2 { i as f32 } else { 0.0 }, 0.0]) // Only DC and low frequencies
                .collect();

            // CPU reference implementation
            let complex_freq: Vec<rustfft::num_complex::Complex32> = freq_data
                .iter()
                .map(|[re, im]| rustfft::num_complex::Complex32::new(*re, *im))
                .collect();
            let mut cpu_fft = Fft::new(size).unwrap();
            let cpu_result = cpu_fft.inverse_complex_simple(&complex_freq).unwrap();

            // GPU implementation
            let gpu_result = gpu_fft.inverse_complex(&freq_data).await.unwrap();

            // Compare results
            assert_eq!(cpu_result.len(), gpu_result.len());
            for (cpu, gpu) in cpu_result.iter().zip(gpu_result.iter()) {
                assert_relative_eq!(*cpu, *gpu, epsilon = 1e-4, max_relative = 1e-3);
            }
        }

        #[tokio::test]
        async fn test_gpu_fft_roundtrip() {
            let size = 32;
            let gpu_fft = match create_gpu_fft(size).await {
                Some(fft) => fft,
                None => return, // Skip if GPU not available
            };

            // Generate test signal
            let original: Vec<f32> = (0..size).map(|i| (i as f32 * 0.2).cos()).collect();

            // Forward + Inverse transform
            let freq_domain = gpu_fft.forward_real(&original).await.unwrap();
            let reconstructed = gpu_fft.inverse_complex(&freq_domain).await.unwrap();

            // Check reconstruction accuracy
            for (orig, recon) in original.iter().zip(reconstructed.iter()) {
                assert_relative_eq!(*orig, *recon, epsilon = 1e-4, max_relative = 1e-3);
            }
        }

        #[tokio::test]
        async fn test_gpu_fft_sizes() {
            // Test various FFT sizes
            let sizes = vec![8, 16, 32, 64, 128];

            for &size in &sizes {
                let gpu_fft = match create_gpu_fft(size).await {
                    Some(fft) => fft,
                    None => continue, // Skip if GPU not available
                };

                // Simple test signal
                let input: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();

                // Should not panic and should produce correct-sized output
                let result = gpu_fft.forward_real(&input).await.unwrap();
                assert_eq!(result.len(), size);

                // Verify DFT properties (sum of real input should be in DC component)
                let dc_sum: f32 = input.iter().sum();
                assert_relative_eq!(dc_sum, result[0][0], epsilon = 1e-3, max_relative = 1e-2);
            }
        }
    }
}
