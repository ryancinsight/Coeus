//! Apollo-backed 3-D FFT kernels for Coeus tensors.
//!
//! The forward transform routes a contiguous row-major `[nx, ny, nz]` signal
//! through [`apollo_fft::fft_3d_leto`]; the inverse routes the full complex
//! spectrum through [`apollo_fft::ifft_3d_leto`]. `fft_3d` computes the
//! unnormalized 3-D DFT, `ifft_3d` divides by `nx * ny * nz`, so
//! `ifft_3d(fft_3d(x)) == x` up to floating-point rounding.
//!
//! Placement matches the 1-D [`crate::fft`] and 2-D [`crate::fft2d`]
//! wrappers: the strided tensor is materialized contiguous on the host, the
//! Apollo transform runs there, and the spectrum is copied back into
//! provider storage. No new `BackendOp` opcode, no provider dispatch change.

use coeus_core::{Complex, ComputeBackend, Float, FloatElement};
use coeus_tensor::Tensor;
use std::ops::Neg;

/// Scalar types supported by Apollo-backed Coeus 3-D FFT operations.
///
/// `f32`/`f64` only: the Apollo plan traits are implemented for the native
/// precisions, not the reduced-storage wrappers. Each impl routes through
/// the Apollo Leto bridge for its own precision, so the tensor-level
/// wrappers below stay free of provider-trait bounds.
pub trait Fft3DScalar: Float + FloatElement + Neg<Output = Self> {
    /// Compute a 3-D forward FFT for a contiguous row-major `[nx, ny, nz]` signal.
    fn fft_3d_impl(signal: &[Self], nx: usize, ny: usize, nz: usize) -> Vec<Complex<Self>>;

    /// Compute a 3-D inverse FFT for a contiguous row-major `[nx, ny, nz]` spectrum.
    fn ifft_3d_impl(spectrum: &[Complex<Self>], nx: usize, ny: usize, nz: usize) -> Vec<Self>;
}

/// View three contiguous row-major lanes as a Leto 3-D array view.
///
/// The caller guarantees `data.len() == nx * ny * nz` and C-order; the
/// layout constructor validates the shape before the view is built.
fn as_array_view<T>(data: &[T], nx: usize, ny: usize, nz: usize) -> leto::ArrayView3<'_, T> {
    let layout = leto::Layout::<3>::c_contiguous([nx, ny, nz])
        .expect("invariant: [nx, ny, nz] from a live tensor shape is a valid layout");
    leto::ArrayView3::try_new(layout, data)
        .expect("invariant: contiguous host buffer matches [nx, ny, nz]")
}

impl Fft3DScalar for f32 {
    #[inline]
    fn fft_3d_impl(signal: &[Self], nx: usize, ny: usize, nz: usize) -> Vec<Complex<Self>> {
        let spectrum = apollo_fft::fft_3d_leto(as_array_view(signal, nx, ny, nz));
        spectrum
            .as_slice()
            .expect("invariant: Apollo 3-D spectrum is contiguous")
            .to_vec()
    }

    #[inline]
    fn ifft_3d_impl(spectrum: &[Complex<Self>], nx: usize, ny: usize, nz: usize) -> Vec<Self> {
        let signal = apollo_fft::ifft_3d_leto::<f32>(as_array_view(spectrum, nx, ny, nz));
        signal
            .as_slice()
            .expect("invariant: Apollo 3-D signal is contiguous")
            .to_vec()
    }
}

impl Fft3DScalar for f64 {
    #[inline]
    fn fft_3d_impl(signal: &[Self], nx: usize, ny: usize, nz: usize) -> Vec<Complex<Self>> {
        let spectrum = apollo_fft::fft_3d_leto(as_array_view(signal, nx, ny, nz));
        spectrum
            .as_slice()
            .expect("invariant: Apollo 3-D spectrum is contiguous")
            .to_vec()
    }

    #[inline]
    fn ifft_3d_impl(spectrum: &[Complex<Self>], nx: usize, ny: usize, nz: usize) -> Vec<Self> {
        let signal = apollo_fft::ifft_3d_leto::<f64>(as_array_view(spectrum, nx, ny, nz));
        signal
            .as_slice()
            .expect("invariant: Apollo 3-D signal is contiguous")
            .to_vec()
    }
}

/// Apollo-backed 3-D forward FFT for Coeus tensors.
///
/// The input must be 3-D with shape `[nx, ny, nz]`; the output spectrum has
/// the same shape. Unnormalized forward 3-D DFT.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn fft_3d<T, B>(signal: &Tensor<T, B>) -> Result<Tensor<Complex<T>, B>, B::Error>
where
    T: Fft3DScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(signal.ndim(), 3, "fft_3d requires 3-D input");
    let backend = B::default();
    let shape = signal.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let input = crate::fft::tensor_to_vec(signal)?;
    let spectrum = T::fft_3d_impl(&input, nx, ny, nz);
    Tensor::from_slice_on(vec![nx, ny, nz], &spectrum, &backend)
}

/// Apollo-backed 3-D inverse FFT for Coeus tensors.
///
/// The input must be a 3-D full complex spectrum with shape `[nx, ny, nz]`;
/// the output signal has the same shape. Divides by `nx * ny * nz`.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn ifft_3d<T, B>(spectrum: &Tensor<Complex<T>, B>) -> Result<Tensor<T, B>, B::Error>
where
    T: Fft3DScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(spectrum.ndim(), 3, "ifft_3d requires 3-D input");
    let backend = B::default();
    let shape = spectrum.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let input = crate::fft::tensor_to_vec(spectrum)?;
    let signal = T::ifft_3d_impl(&input, nx, ny, nz);
    Tensor::from_slice_on(vec![nx, ny, nz], &signal, &backend)
}
