//! Apollo-backed 2-D FFT kernels for Coeus tensors.
//!
//! The forward transform routes a contiguous row-major `[nx, ny]` signal
//! through [`apollo_fft::fft_2d_leto`]; the inverse routes the full complex
//! spectrum through [`apollo_fft::ifft_2d_leto`]. `fft_2d` computes the
//! unnormalized 2-D DFT, `ifft_2d` divides by `nx * ny`, so
//! `ifft_2d(fft_2d(x)) == x` up to floating-point rounding.
//!
//! Placement matches the 1-D [`crate::fft`] wrappers: the strided tensor is
//! materialized contiguous on the host, the Apollo transform runs there, and
//! the spectrum is copied back into provider storage. No new `BackendOp`
//! opcode, no provider dispatch change.

use coeus_core::{Complex, ComputeBackend, Float, FloatElement};
use coeus_tensor::Tensor;
use std::ops::Neg;

/// Scalar types supported by Apollo-backed Coeus 2-D FFT operations.
///
/// `f32`/`f64` only: the Apollo plan traits are implemented for the native
/// precisions, not the reduced-storage wrappers. Each impl routes through
/// the Apollo Leto bridge for its own precision, so the tensor-level
/// wrappers below stay free of provider-trait bounds.
pub trait Fft2DScalar: Float + FloatElement + Neg<Output = Self> {
    /// Compute a 2-D forward FFT for a contiguous row-major `[nx, ny]` signal.
    fn fft_2d_impl(signal: &[Self], nx: usize, ny: usize) -> Vec<Complex<Self>>;

    /// Compute a 2-D inverse FFT for a contiguous row-major `[nx, ny]` spectrum.
    fn ifft_2d_impl(spectrum: &[Complex<Self>], nx: usize, ny: usize) -> Vec<Self>;
}

/// View two contiguous row-major lanes as a Leto 2-D array view.
///
/// The caller guarantees `data.len() == nx * ny` and C-order; the layout
/// constructor validates the shape before the view is built.
fn as_array_view<T>(data: &[T], nx: usize, ny: usize) -> leto::ArrayView2<'_, T> {
    let layout = leto::Layout::<2>::c_contiguous([nx, ny])
        .expect("invariant: [nx, ny] from a live tensor shape is a valid layout");
    leto::ArrayView2::try_new(layout, data)
        .expect("invariant: contiguous host buffer matches [nx, ny]")
}

impl Fft2DScalar for f32 {
    #[inline]
    fn fft_2d_impl(signal: &[Self], nx: usize, ny: usize) -> Vec<Complex<Self>> {
        let spectrum = apollo_fft::fft_2d_leto(as_array_view(signal, nx, ny));
        spectrum
            .as_slice()
            .expect("invariant: Apollo 2-D spectrum is contiguous")
            .to_vec()
    }

    #[inline]
    fn ifft_2d_impl(spectrum: &[Complex<Self>], nx: usize, ny: usize) -> Vec<Self> {
        let signal = apollo_fft::ifft_2d_leto::<f32>(as_array_view(spectrum, nx, ny));
        signal
            .as_slice()
            .expect("invariant: Apollo 2-D signal is contiguous")
            .to_vec()
    }
}

impl Fft2DScalar for f64 {
    #[inline]
    fn fft_2d_impl(signal: &[Self], nx: usize, ny: usize) -> Vec<Complex<Self>> {
        let spectrum = apollo_fft::fft_2d_leto(as_array_view(signal, nx, ny));
        spectrum
            .as_slice()
            .expect("invariant: Apollo 2-D spectrum is contiguous")
            .to_vec()
    }

    #[inline]
    fn ifft_2d_impl(spectrum: &[Complex<Self>], nx: usize, ny: usize) -> Vec<Self> {
        let signal = apollo_fft::ifft_2d_leto::<f64>(as_array_view(spectrum, nx, ny));
        signal
            .as_slice()
            .expect("invariant: Apollo 2-D signal is contiguous")
            .to_vec()
    }
}

/// Apollo-backed 2-D forward FFT for Coeus tensors.
///
/// The input must be 2-D with shape `[nx, ny]`; the output spectrum has the
/// same shape. Unnormalized forward 2-D DFT.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn fft_2d<T, B>(signal: &Tensor<T, B>) -> Result<Tensor<Complex<T>, B>, B::Error>
where
    T: Fft2DScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(signal.ndim(), 2, "fft_2d requires 2-D input");
    let backend = B::default();
    let shape = signal.shape();
    let (nx, ny) = (shape[0], shape[1]);
    let input = crate::fft::tensor_to_vec(signal)?;
    let spectrum = T::fft_2d_impl(&input, nx, ny);
    Tensor::from_slice_on(vec![nx, ny], &spectrum, &backend)
}

/// Apollo-backed 2-D inverse FFT for Coeus tensors.
///
/// The input must be a 2-D full complex spectrum with shape `[nx, ny]`; the
/// output signal has the same shape. Divides by `nx * ny`.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn ifft_2d<T, B>(spectrum: &Tensor<Complex<T>, B>) -> Result<Tensor<T, B>, B::Error>
where
    T: Fft2DScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(spectrum.ndim(), 2, "ifft_2d requires 2-D input");
    let backend = B::default();
    let shape = spectrum.shape();
    let (nx, ny) = (shape[0], shape[1]);
    let input = crate::fft::tensor_to_vec(spectrum)?;
    let signal = T::ifft_2d_impl(&input, nx, ny);
    Tensor::from_slice_on(vec![nx, ny], &signal, &backend)
}
