//! Apollo-backed half-spectrum 2-D and 3-D FFT kernels for Coeus tensors.
//!
//! A real field's spectrum is Hermitian-redundant, so the last axis keeps
//! only `n/2 + 1` bins: `[nx, ny/2 + 1]` and `[nx, ny, nz/2 + 1]`.
//! The forward entries route through Apollo's true half-plane kernels
//! (`fft_{2,3}d_array_half_into`); the inverse entries invert through
//! `ifft_{2,3}d_array_half_into` with the full `1/(nx·ny[·nz])`
//! normalization, so `irfft(rfft(x)) == x` up to floating-point rounding.
//!
//! Unlike the 1-D pair (which mirrors through the full transform), these
//! entries use Apollo's true half-plane kernels — no redundant half is ever
//! computed. Placement otherwise matches the other wrappers: the strided
//! tensor is materialized contiguous on the host, the Apollo transform runs
//! there, and the result is copied back into provider storage. No new
//! `BackendOp` opcode, no provider dispatch change.

use coeus_core::{Complex, ComputeBackend, Float, FloatElement};
use coeus_tensor::Tensor;
use std::ops::Neg;

/// Scalar types supported by the Apollo-backed multidimensional
/// half-spectrum FFT operations.
///
/// `f32`/`f64` only: the Apollo plan traits are implemented for the native
/// precisions, not the reduced-storage wrappers. Each impl routes through
/// the Apollo half-plane kernels for its own precision, so the
/// tensor-level wrappers below stay free of provider-trait bounds —
/// exactly the `FftScalar`/`Fft2DScalar` pattern.
pub trait FftHalfNdScalar: Float + FloatElement + Neg<Output = Self> {
    /// Compute the `[nx, ny/2 + 1]` bins of a 2-D real field's forward FFT.
    fn rfft_2d_impl(signal: &[Self], nx: usize, ny: usize) -> Vec<Complex<Self>>;

    /// Invert a `[nx, ny/2 + 1]` half spectrum into `[nx, ny]` reals.
    fn irfft_2d_impl(spectrum: &[Complex<Self>], nx: usize, ny: usize) -> Vec<Self>;

    /// Compute the `[nx, ny, nz/2 + 1]` bins of a 3-D real field's forward FFT.
    fn rfft_3d_impl(signal: &[Self], nx: usize, ny: usize, nz: usize) -> Vec<Complex<Self>>;

    /// Invert a `[nx, ny, nz/2 + 1]` half spectrum into `[nx, ny, nz]` reals.
    fn irfft_3d_impl(
        spectrum: &[Complex<Self>],
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Vec<Self>;
}

/// Build a C-contiguous Leto `Array2` from a host buffer.
fn array_2d<T: Clone>(data: Vec<T>, nx: usize, ny: usize) -> leto::Array2<T> {
    let layout = leto::Layout::<2>::c_contiguous([nx, ny])
        .expect("invariant: [nx, ny] from a live tensor shape is a valid layout");
    leto::Array::<T, leto::VecStorage<T>, 2>::new(layout, leto::VecStorage::new(data))
        .expect("invariant: contiguous host buffer matches [nx, ny]")
}

/// Build a C-contiguous Leto `Array3` from a host buffer.
fn array_3d<T: Clone>(data: Vec<T>, nx: usize, ny: usize, nz: usize) -> leto::Array3<T> {
    let layout = leto::Layout::<3>::c_contiguous([nx, ny, nz])
        .expect("invariant: [nx, ny, nz] from a live tensor shape is a valid layout");
    leto::Array::<T, leto::VecStorage<T>, 3>::new(layout, leto::VecStorage::new(data))
        .expect("invariant: contiguous host buffer matches [nx, ny, nz]")
}

impl FftHalfNdScalar for f32 {
    #[inline]
    fn rfft_2d_impl(signal: &[Self], nx: usize, ny: usize) -> Vec<Complex<Self>> {
        let field = array_2d(signal.to_vec(), nx, ny);
        let half_ny = ny / 2 + 1;
        let mut out = array_2d(vec![Complex::<Self>::default(); nx * half_ny], nx, half_ny);
        apollo_fft::fft_2d_array_half_into(&field, &mut out);
        out.as_slice()
            .expect("invariant: Apollo 2-D half spectrum is contiguous")
            .to_vec()
    }

    #[inline]
    fn irfft_2d_impl(spectrum: &[Complex<Self>], nx: usize, ny: usize) -> Vec<Self> {
        let half_ny = ny / 2 + 1;
        let mut field_hat = array_2d(spectrum.to_vec(), nx, half_ny);
        let mut out = array_2d(vec![0.0; nx * ny], nx, ny);
        apollo_fft::ifft_2d_array_half_into(&mut field_hat, &mut out);
        out.as_slice()
            .expect("invariant: Apollo 2-D signal is contiguous")
            .to_vec()
    }

    #[inline]
    fn rfft_3d_impl(signal: &[Self], nx: usize, ny: usize, nz: usize) -> Vec<Complex<Self>> {
        let field = array_3d(signal.to_vec(), nx, ny, nz);
        let half_nz = nz / 2 + 1;
        let mut out = array_3d(
            vec![Complex::<Self>::default(); nx * ny * half_nz],
            nx,
            ny,
            half_nz,
        );
        apollo_fft::fft_3d_array_half_into(&field, &mut out);
        out.as_slice()
            .expect("invariant: Apollo 3-D half spectrum is contiguous")
            .to_vec()
    }

    #[inline]
    fn irfft_3d_impl(
        spectrum: &[Complex<Self>],
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Vec<Self> {
        let half_nz = nz / 2 + 1;
        let mut field_hat = array_3d(spectrum.to_vec(), nx, ny, half_nz);
        let mut out = array_3d(vec![0.0; nx * ny * nz], nx, ny, nz);
        apollo_fft::ifft_3d_array_half_into(&mut field_hat, &mut out);
        out.as_slice()
            .expect("invariant: Apollo 3-D signal is contiguous")
            .to_vec()
    }
}

impl FftHalfNdScalar for f64 {
    #[inline]
    fn rfft_2d_impl(signal: &[Self], nx: usize, ny: usize) -> Vec<Complex<Self>> {
        let field = array_2d(signal.to_vec(), nx, ny);
        let half_ny = ny / 2 + 1;
        let mut out = array_2d(vec![Complex::<Self>::default(); nx * half_ny], nx, half_ny);
        apollo_fft::fft_2d_array_half_into(&field, &mut out);
        out.as_slice()
            .expect("invariant: Apollo 2-D half spectrum is contiguous")
            .to_vec()
    }

    #[inline]
    fn irfft_2d_impl(spectrum: &[Complex<Self>], nx: usize, ny: usize) -> Vec<Self> {
        let half_ny = ny / 2 + 1;
        let mut field_hat = array_2d(spectrum.to_vec(), nx, half_ny);
        let mut out = array_2d(vec![0.0; nx * ny], nx, ny);
        apollo_fft::ifft_2d_array_half_into(&mut field_hat, &mut out);
        out.as_slice()
            .expect("invariant: Apollo 2-D signal is contiguous")
            .to_vec()
    }

    #[inline]
    fn rfft_3d_impl(signal: &[Self], nx: usize, ny: usize, nz: usize) -> Vec<Complex<Self>> {
        let field = array_3d(signal.to_vec(), nx, ny, nz);
        let half_nz = nz / 2 + 1;
        let mut out = array_3d(
            vec![Complex::<Self>::default(); nx * ny * half_nz],
            nx,
            ny,
            half_nz,
        );
        apollo_fft::fft_3d_array_half_into(&field, &mut out);
        out.as_slice()
            .expect("invariant: Apollo 3-D half spectrum is contiguous")
            .to_vec()
    }

    #[inline]
    fn irfft_3d_impl(
        spectrum: &[Complex<Self>],
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Vec<Self> {
        let half_nz = nz / 2 + 1;
        let mut field_hat = array_3d(spectrum.to_vec(), nx, ny, half_nz);
        let mut out = array_3d(vec![0.0; nx * ny * nz], nx, ny, nz);
        apollo_fft::ifft_3d_array_half_into(&mut field_hat, &mut out);
        out.as_slice()
            .expect("invariant: Apollo 3-D signal is contiguous")
            .to_vec()
    }
}

/// Hermitian completion of a 2-D half spectrum to full length.
///
/// `full[k1,k2] = half[k1,k2]` for `k2 <= ny/2`, else
/// `conj(half[(nx-k1)%nx, ny-k2])`. Panics when the half shape is not
/// `[nx, ny/2 + 1]`.
pub(crate) fn mirror_complete_2d<T: Float + Neg<Output = T> + Copy>(
    half: &[Complex<T>],
    nx: usize,
    ny: usize,
) -> Vec<Complex<T>> {
    let half_ny = ny / 2 + 1;
    assert_eq!(
        half.len(),
        nx * half_ny,
        "mirror_complete_2d: half spectrum must hold nx * (ny/2 + 1) bins"
    );
    let at = |data: &[Complex<T>], i1: usize, i2: usize, stride: usize| data[i1 * stride + i2];
    let mut full = vec![Complex::new(T::ZERO, T::ZERO); nx * ny];
    for k1 in 0..nx {
        for k2 in 0..ny {
            let value = if k2 <= ny / 2 {
                at(half, k1, k2, half_ny)
            } else {
                let source = at(half, (nx - k1) % nx, ny - k2, half_ny);
                Complex::new(source.re, -source.im)
            };
            full[k1 * ny + k2] = value;
        }
    }
    full
}

/// Fold a full 2-D adjoint into half-spectrum lanes.
///
/// The completion adjoint: `half[k] = (F[k] + conj(F[mirror(k)])) / n` for
/// lanes whose mirror lies outside the half set, `F[k] / n` for the
/// self-mirror corners (`k1` in `{0, nx/2}`, `k2` in `{0, ny/2}`).
pub(crate) fn fold_half_2d<T: Float + Neg<Output = T> + Copy>(
    full: &[Complex<T>],
    nx: usize,
    ny: usize,
    scale: T,
) -> Vec<Complex<T>> {
    let half_ny = ny / 2 + 1;
    assert_eq!(
        full.len(),
        nx * ny,
        "fold_half_2d: full adjoint must hold nx * ny bins"
    );
    let at = |data: &[Complex<T>], i1: usize, i2: usize| data[i1 * ny + i2];
    let mut half = Vec::with_capacity(nx * half_ny);
    for k1 in 0..nx {
        for k2 in 0..half_ny {
            let self_mirror = (k1 == 0 || (nx % 2 == 0 && k1 * 2 == nx))
                && (k2 == 0 || (ny % 2 == 0 && k2 * 2 == ny));
            let value = at(full, k1, k2);
            let folded = if self_mirror {
                value
            } else {
                let mirror = at(full, (nx - k1) % nx, (ny - k2) % ny);
                Complex::new(value.re + mirror.re, value.im - mirror.im)
            };
            half.push(Complex::new(folded.re / scale, folded.im / scale));
        }
    }
    half
}

/// Hermitian completion of a 3-D half spectrum to full length.
///
/// `full[k1,k2,k3] = half[k1,k2,k3]` for `k3 <= nz/2`, else
/// `conj(half[(nx-k1)%nx, (ny-k2)%ny, nz-k3])`.
pub(crate) fn mirror_complete_3d<T: Float + Neg<Output = T> + Copy>(
    half: &[Complex<T>],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<Complex<T>> {
    let half_nz = nz / 2 + 1;
    assert_eq!(
        half.len(),
        nx * ny * half_nz,
        "mirror_complete_3d: half spectrum must hold nx * ny * (nz/2 + 1) bins"
    );
    let at = |data: &[Complex<T>], i1: usize, i2: usize, i3: usize, s2: usize, s3: usize| {
        data[(i1 * s2 + i2) * s3 + i3]
    };
    let mut full = vec![Complex::new(T::ZERO, T::ZERO); nx * ny * nz];
    for k1 in 0..nx {
        for k2 in 0..ny {
            for k3 in 0..nz {
                let value = if k3 <= nz / 2 {
                    at(half, k1, k2, k3, ny, half_nz)
                } else {
                    let source = at(half, (nx - k1) % nx, (ny - k2) % ny, nz - k3, ny, half_nz);
                    Complex::new(source.re, -source.im)
                };
                full[(k1 * ny + k2) * nz + k3] = value;
            }
        }
    }
    full
}

/// Fold a full 3-D adjoint into half-spectrum lanes.
///
/// Same completion-adjoint as [`fold_half_2d`]: self-mirror corners
/// (`k1` in `{0, nx/2}`, `k2` in `{0, ny/2}`, `k3` in `{0, nz/2}`) fold
/// singly, every other lane adds its mirror's conjugate.
pub(crate) fn fold_half_3d<T: Float + Neg<Output = T> + Copy>(
    full: &[Complex<T>],
    nx: usize,
    ny: usize,
    nz: usize,
    scale: T,
) -> Vec<Complex<T>> {
    let half_nz = nz / 2 + 1;
    assert_eq!(
        full.len(),
        nx * ny * nz,
        "fold_half_3d: full adjoint must hold nx * ny * nz bins"
    );
    let at = |data: &[Complex<T>], i1: usize, i2: usize, i3: usize| {
        data[(i1 * ny + i2) * nz + i3]
    };
    let mut half = Vec::with_capacity(nx * ny * half_nz);
    for k1 in 0..nx {
        for k2 in 0..ny {
            for k3 in 0..half_nz {
                let self_mirror = (k1 == 0 || (nx % 2 == 0 && k1 * 2 == nx))
                    && (k2 == 0 || (ny % 2 == 0 && k2 * 2 == ny))
                    && (k3 == 0 || (nz % 2 == 0 && k3 * 2 == nz));
                let value = at(full, k1, k2, k3);
                let folded = if self_mirror {
                    value
                } else {
                    let mirror = at(full, (nx - k1) % nx, (ny - k2) % ny, (nz - k3) % nz);
                    Complex::new(value.re + mirror.re, value.im - mirror.im)
                };
                half.push(Complex::new(folded.re / scale, folded.im / scale));
            }
        }
    }
    half
}
///
/// The input must be 2-D with shape `[nx, ny]`; the output holds the
/// `[nx, ny/2 + 1]` independent bins. Unnormalized.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn rfft_2d<T, B>(signal: &Tensor<T, B>) -> Result<Tensor<Complex<T>, B>, B::Error>
where
    T: FftHalfNdScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(signal.ndim(), 2, "rfft_2d requires 2-D input");
    let backend = B::default();
    let shape = signal.shape();
    let (nx, ny) = (shape[0], shape[1]);
    let input = crate::fft::tensor_to_vec(signal)?;
    let spectrum = T::rfft_2d_impl(&input, nx, ny);
    Tensor::from_slice_on(vec![nx, ny / 2 + 1], &spectrum, &backend)
}

/// Apollo-backed half-spectrum 2-D inverse FFT for Coeus tensors.
///
/// The input must be a 2-D half spectrum with shape `[nx, ny/2 + 1]`; the
/// output is `[nx, ny]` reals. Divides by `nx * ny`.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn irfft_2d<T, B>(
    spectrum: &Tensor<Complex<T>, B>,
    nx: usize,
    ny: usize,
) -> Result<Tensor<T, B>, B::Error>
where
    T: FftHalfNdScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(spectrum.ndim(), 2, "irfft_2d requires 2-D input");
    let half_ny = ny / 2 + 1;
    assert!(
        spectrum.shape() == &[nx, half_ny],
        "irfft_2d: half spectrum must have shape [nx, ny/2 + 1]"
    );
    let backend = B::default();
    let input = crate::fft::tensor_to_vec(spectrum)?;
    let signal = T::irfft_2d_impl(&input, nx, ny);
    Tensor::from_slice_on(vec![nx, ny], &signal, &backend)
}

/// Apollo-backed half-spectrum 3-D forward FFT for Coeus tensors.
///
/// The input must be 3-D with shape `[nx, ny, nz]`; the output holds the
/// `[nx, ny, nz/2 + 1]` independent bins. Unnormalized.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn rfft_3d<T, B>(signal: &Tensor<T, B>) -> Result<Tensor<Complex<T>, B>, B::Error>
where
    T: FftHalfNdScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(signal.ndim(), 3, "rfft_3d requires 3-D input");
    let backend = B::default();
    let shape = signal.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let input = crate::fft::tensor_to_vec(signal)?;
    let spectrum = T::rfft_3d_impl(&input, nx, ny, nz);
    Tensor::from_slice_on(vec![nx, ny, nz / 2 + 1], &spectrum, &backend)
}

/// Apollo-backed half-spectrum 3-D inverse FFT for Coeus tensors.
///
/// The input must be a 3-D half spectrum with shape `[nx, ny, nz/2 + 1]`;
/// the output is `[nx, ny, nz]` reals. Divides by `nx * ny * nz`.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn irfft_3d<T, B>(
    spectrum: &Tensor<Complex<T>, B>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<Tensor<T, B>, B::Error>
where
    T: FftHalfNdScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(spectrum.ndim(), 3, "irfft_3d requires 3-D input");
    let half_nz = nz / 2 + 1;
    assert!(
        spectrum.shape() == &[nx, ny, half_nz],
        "irfft_3d: half spectrum must have shape [nx, ny, nz/2 + 1]"
    );
    let backend = B::default();
    let input = crate::fft::tensor_to_vec(spectrum)?;
    let signal = T::irfft_3d_impl(&input, nx, ny, nz);
    Tensor::from_slice_on(vec![nx, ny, nz], &signal, &backend)
}
