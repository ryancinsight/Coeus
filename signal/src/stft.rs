//! Short-Time Fourier Transform (STFT).

use crate::windows::WindowFunc;
use backend::CpuBackend;
use coeus_error::{Error, Result, StorageError, TensorError};
use dtype::complex::Complex32;
use dtype::float::Float32;
use fft::CpuFft;
use storage::DenseStorage;
use storage::Storage;
use tensor::Tensor;

/// STFT implementation
pub trait STFT {
    fn stft(
        &self,
        n_fft: usize,
        hop_length: Option<usize>,
        win_length: Option<usize>,
        window: Option<&Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>,
        center: bool,
    ) -> Result<Tensor<CpuBackend<Complex32>, DenseStorage<Complex32>, Complex32>>;
}

impl STFT for Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> {
    fn stft(
        &self,
        n_fft: usize,
        hop_length: Option<usize>,
        win_length: Option<usize>,
        window: Option<&Self>,
        center: bool,
    ) -> Result<Tensor<CpuBackend<Complex32>, DenseStorage<Complex32>, Complex32>> {
        let signal = self.as_slice();
        let hop_length = hop_length.unwrap_or(n_fft / 4);
        let win_length = win_length.unwrap_or(n_fft);

        if win_length > n_fft {
            return Err(Error::Tensor(TensorError::ShapeMismatch(
                "win_length must be <= n_fft".to_string(),
            )));
        }

        // 1. Padding if center is true
        let padded_signal = if center {
            let pad_size = n_fft / 2;
            let mut padded = Vec::with_capacity(signal.len() + 2 * pad_size);
            // Reflect padding (simplified: zero padding for now, or symmetric)
            for _ in 0..pad_size {
                padded.push(Float32::new(0.0));
            }
            padded.extend_from_slice(signal);
            for _ in 0..pad_size {
                padded.push(Float32::new(0.0));
            }
            padded
        } else {
            signal.to_vec()
        };

        // 2. Prepare FFT
        let fft = CpuFft::new(n_fft);

        // 3. Framing and Windowing
        let num_frames = (padded_signal.len() - win_length) / hop_length + 1;
        let mut stft_data = Vec::with_capacity(num_frames * n_fft);

        // Default window if None
        let default_window = if window.is_none() {
            Some(Self::hann_window(win_length, false)?)
        } else {
            None
        };

        let win_tensor = match window {
            Some(win) => win,
            None => default_window.as_ref().unwrap(),
        };
        let win_data = win_tensor.as_slice();

        for i in 0..num_frames {
            let start = i * hop_length;
            let mut frame = vec![Float32::new(0.0); n_fft];

            // Extract and window
            for j in 0..win_length {
                frame[j] = Float32::new(padded_signal[start + j].get() * win_data[j].get());
            }

            // Perform FFT
            let frame_storage = DenseStorage::from_vec(frame, &[n_fft])
                .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
            let spec_storage = fft.forward(&frame_storage)?;

            stft_data.extend_from_slice(spec_storage.as_slice());
        }

        // Output shape: [num_frames, n_fft]
        let storage = DenseStorage::from_vec(stft_data, &[num_frames, n_fft])
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;

        Ok(Tensor::from_storage(
            storage,
            CpuBackend::<Complex32>::new(),
        ))
    }
}
