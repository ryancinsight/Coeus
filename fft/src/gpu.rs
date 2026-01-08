//! GPU-based FFT implementation using wgpu shaders

use backend::gpu::GpuBackend;
use coeus_error::Result;
use storage::DenseStorage;
use storage::Storage;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuComplex32 {
    re: f32,
    im: f32,
}

/// GPU FFT processor using wgpu shaders from the backend
pub struct GpuFft {
    backend: GpuBackend<dtype::float::Float32>,
    size: usize,
}

impl GpuFft {
    /// Create a new GPU FFT processor
    pub fn new(backend: GpuBackend<dtype::float::Float32>, size: usize) -> Self {
        Self { backend, size }
    }

    /// Perform forward FFT (Complex -> Complex)
    pub fn fft(
        &self,
        input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        self.run_fft_complex(input, false)
    }

    /// Perform inverse FFT (Complex -> Complex)
    pub fn ifft(
        &self,
        input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        self.run_fft_complex(input, true)
    }

    /// Perform Real-to-Complex FFT (Float -> Complex, one-sided)
    pub fn rfft(
        &self,
        input: &DenseStorage<dtype::float::Float32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        let full = self.run_fft_real_to_complex(input, false)?;

        // One-sided spectrum: size/2 + 1
        let out_size = self.size / 2 + 1;
        let data = full.as_slice();
        DenseStorage::from_vec(data[..out_size].to_vec(), &[out_size])
            .map_err(|e| coeus_error::Error::from(coeus_error::StorageError::from(e.to_string())))
    }

    /// Perform Complex-to-Real Inverse FFT (Complex -> Float, one-sided)
    pub fn irfft(
        &self,
        input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::float::Float32>> {
        let one_sided = input.as_slice();
        let expected_input_size = self.size / 2 + 1;
        if one_sided.len() < expected_input_size {
            return Err(coeus_error::TensorError::ShapeMismatch(format!(
                "IRFFT input too short: expected at least {}, got {}",
                expected_input_size,
                one_sided.len()
            ))
            .into());
        }

        // Reconstruct full spectrum for GPU
        let mut full_spectrum = vec![dtype::complex::Complex32::new(0.0, 0.0); self.size];
        full_spectrum[0] = one_sided[0];
        for i in 1..expected_input_size - 1 {
            full_spectrum[i] = one_sided[i];
            full_spectrum[self.size - i] = one_sided[i].conj();
        }
        if self.size % 2 == 0 {
            full_spectrum[self.size / 2] = one_sided[expected_input_size - 1];
        }

        let full_storage = DenseStorage::from_vec(full_spectrum, &[self.size]).map_err(|e| {
            coeus_error::Error::from(coeus_error::StorageError::from(e.to_string()))
        })?;

        let result_complex = self.run_fft_complex(&full_storage, true)?;

        // Return real part
        let float_data: Vec<dtype::float::Float32> = result_complex
            .as_slice()
            .iter()
            .map(|c| dtype::float::Float32::new(c.re))
            .collect();

        DenseStorage::from_vec(float_data, &[self.size])
            .map_err(|e| coeus_error::Error::from(coeus_error::StorageError::from(e.to_string())))
    }

    /// Legacy forward compatibility (Full spectrum)
    pub fn forward(
        &self,
        input: &DenseStorage<dtype::float::Float32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        self.run_fft_real_to_complex(input, false)
    }

    /// Legacy inverse compatibility
    pub fn inverse(
        &self,
        input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::float::Float32>> {
        let result_complex = self.run_fft_complex(input, true)?;
        let float_data: Vec<dtype::float::Float32> = result_complex
            .as_slice()
            .iter()
            .map(|c| dtype::float::Float32::new(c.re))
            .collect();
        DenseStorage::from_vec(float_data, &[self.size])
            .map_err(|e| coeus_error::Error::from(coeus_error::StorageError::from(e.to_string())))
    }

    fn run_fft_complex(
        &self,
        input: &DenseStorage<dtype::complex::Complex32>,
        inverse: bool,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        let input_data = input.as_slice();
        let complex_gpu_data: Vec<GpuComplex32> = input_data
            .iter()
            .take(self.size)
            .map(|c| GpuComplex32 { re: c.re, im: c.im })
            .collect();

        self.execute_gpu_fft(complex_gpu_data, inverse)
    }

    fn run_fft_real_to_complex(
        &self,
        input: &DenseStorage<dtype::float::Float32>,
        inverse: bool,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        let input_data = input.as_slice();
        let mut complex_data = vec![GpuComplex32 { re: 0.0, im: 0.0 }; self.size];
        for (i, &val) in input_data.iter().take(self.size).enumerate() {
            complex_data[i] = GpuComplex32 {
                re: val.get(),
                im: 0.0,
            };
        }
        self.execute_gpu_fft(complex_data, inverse)
    }

    fn execute_gpu_fft(
        &self,
        complex_data: Vec<GpuComplex32>,
        inverse: bool,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        let device = self.backend.wgpu_device();
        let queue = self.backend.wgpu_queue();
        let pipeline = self.backend.fft_pipeline();
        let layout = self.backend.fft_bind_group_layout();

        let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT Data Buffer"),
            contents: bytemuck::cast_slice(&complex_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let inv_flag: u32 = if inverse { 1 } else { 0 };
        let inv_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT Inverse Flag"),
            contents: bytemuck::bytes_of(&inv_flag),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let num_passes = (self.size as f32).log2() as u32;

        for pass in 0..num_passes {
            let params = [self.size as u32, 2u32, pass];
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("FFT Params Pass {}", pass)),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("FFT Bind Group Pass {}", pass)),
                layout,
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
                        resource: inv_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("FFT Pass {} Encoder", pass)),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("FFT Pass {} Compute", pass)),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let workgroups = (self.size as u32 + 255) / 256;
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }

            queue.submit(Some(encoder.finish()));
        }

        let result_complex = self.read_complex_buffer(&data_buffer)?;
        DenseStorage::from_vec(result_complex, &[self.size])
            .map_err(|e| coeus_error::StorageError::InvalidShape(format!("{e}")).into())
    }

    fn read_complex_buffer(&self, buffer: &wgpu::Buffer) -> Result<Vec<dtype::complex::Complex32>> {
        let size = (self.size * std::mem::size_of::<GpuComplex32>()) as u64;
        let staging_buffer = self
            .backend
            .wgpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("FFT Staging Buffer"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let mut encoder =
            self.backend
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("FFT Read Encoder"),
                });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        self.backend.wgpu_queue().submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });

        self.backend.wgpu_device().poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| coeus_error::BackendError::Gpu("GPU map response dropped".to_string()))?
            .map_err(|e| coeus_error::BackendError::Gpu(format!("GPU map failure: {e:?}")))?;

        let data = buffer_slice.get_mapped_range();
        let raw = bytemuck::cast_slice::<u8, GpuComplex32>(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(raw
            .into_iter()
            .map(|c| dtype::complex::Complex32::new(c.re, c.im))
            .collect())
    }
}
