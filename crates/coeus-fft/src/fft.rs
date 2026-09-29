//! Apollo-backed 1-D FFT kernels and the scalar contract.

use coeus_autograd::GradBuffer;
use coeus_core::{Complex, ComputeBackend, Float, Scalar};
use coeus_tensor::Tensor;
use std::ops::Neg;
use std::sync::Arc;

/// Scalar types supported by Apollo-backed Coeus FFT operations.
pub trait FftScalar: Float + Neg<Output = Self> {
    /// Compute a 1-D forward FFT for a contiguous real signal.
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>>;

    /// Compute a 1-D inverse FFT for a contiguous complex spectrum.
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self>;
}

impl FftScalar for f32 {
    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        apollo_fft::fft_1d_slice::<f32>(signal)
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        apollo_fft::ifft_1d_slice::<f32>(spectrum)
    }
}

impl FftScalar for f64 {
    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        apollo_fft::fft_1d_slice::<f64>(signal)
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        apollo_fft::ifft_1d_slice::<f64>(spectrum)
    }
}

pub(crate) fn accumulate_grad<T, B>(grad: &Arc<GradBuffer<T, B>>, delta: &Tensor<T, B>)
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let mut current = vec![T::zero(); delta.numel()];
    let delta_host = tensor_to_vec(delta);
    let guard = grad.write();
    backend.copy_to_host(guard.storage(), &mut current);
    for (dst, src) in current.iter_mut().zip(delta_host) {
        *dst += src;
    }
    backend.copy_to_device(&current, guard.storage_mut());
}

pub(crate) fn tensor_to_vec<T, B>(tensor: &Tensor<T, B>) -> Vec<T>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let contiguous = tensor.to_contiguous();
    let mut host = vec![T::zero(); contiguous.numel()];
    backend.copy_to_host(contiguous.storage(), &mut host);
    host
}

/// Apollo-backed 1-D forward FFT for Coeus tensors.
#[must_use]
pub fn fft_1d<T, B>(signal: &Tensor<T, B>) -> Tensor<Complex<T>, B>
where
    T: FftScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(signal.ndim(), 1, "fft_1d requires 1-D input");
    let backend = B::default();
    let input = tensor_to_vec(signal);
    let spectrum = T::fft_1d_impl(&input);
    Tensor::from_slice_on(signal.shape_cloned(), &spectrum, &backend)
}

/// Apollo-backed 1-D inverse FFT for Coeus tensors.
#[must_use]
pub fn ifft_1d<T, B>(spectrum: &Tensor<Complex<T>, B>) -> Tensor<T, B>
where
    T: FftScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(spectrum.ndim(), 1, "ifft_1d requires 1-D input");
    let backend = B::default();
    let input = tensor_to_vec(spectrum);
    let signal = T::ifft_1d_impl(&input);
    Tensor::from_slice_on(spectrum.shape_cloned(), &signal, &backend)
}
