//! Tensor-level 1-D FFT for Coeus, routed through the Atlas-owned Apollo FFT library.
//!
//! This crate is the **non-autograd** signal-processing seam: it operates directly on
//! [`Tensor`] values (no [`coeus_autograd::Var`] graph, no reverse-mode bookkeeping) and
//! depends only on `coeus-core`, `coeus-tensor`, and `apollo-fft`. It is the SSOT for the
//! Apollo -> Coeus tensor FFT bridge; the differentiable `Var`-level FFT nodes in
//! `coeus-autograd` build on this primitive rather than re-deriving it (see backlog
//! `G-FFT-CONSOLIDATE`: `coeus-autograd/ops/fft.rs` is scheduled to depend on this crate).
//!
//! No dependency on `rustfft`: all transforms execute through `apollo-fft`, keeping signal
//! processing inside the Atlas ecosystem.
//!
//! # Numerical contract
//! [`fft_1d`] computes the unnormalized forward DFT `X[k] = sum_n x[n] * exp(-2i*pi*k*n/N)`.
//! [`ifft_1d`] applies the `1/N`-normalized inverse, so `ifft_1d(fft_1d(x)) == x` up to
//! floating-point rounding.

// ── Coeus FFT ──
// Tensor-level FFT bridge over Apollo.
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use coeus_core::{Complex, ComputeBackend, Float, Scalar};
use coeus_tensor::Tensor;

/// Scalar types with an Apollo-backed 1-D FFT implementation.
///
/// Implemented for the IEEE-754 binary floating-point types Apollo plans natively.
pub trait FftScalar: Float {
    /// Forward FFT of a contiguous real signal, producing its complex spectrum.
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>>;

    /// Inverse (`1/N`-normalized) FFT of a contiguous complex spectrum.
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self>;
}

impl FftScalar for f32 {
    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        apollo_fft::fft_1d_slice_typed::<f32>(signal)
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        apollo_fft::ifft_1d_slice_typed::<f32>(spectrum)
    }
}

impl FftScalar for f64 {
    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        apollo_fft::fft_1d_slice_typed::<f64>(signal)
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        apollo_fft::ifft_1d_slice_typed::<f64>(spectrum)
    }
}

/// Copy a tensor's logical elements into a contiguous host `Vec` in row-major order.
fn tensor_to_vec<T, B>(tensor: &Tensor<T, B>) -> Vec<T>
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

/// Apollo-backed 1-D forward FFT of a real-valued tensor.
///
/// # Arguments
/// * `signal` — 1-D tensor of shape `[N]`.
///
/// # Returns
/// Complex tensor of shape `[N]` holding the frequency spectrum.
///
/// # Panics
/// If `signal` is not 1-D.
///
/// # Examples
/// ```
/// use coeus_fft::fft_1d;
/// use coeus_tensor::Tensor;
///
/// let signal = Tensor::<f64>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
/// let spectrum = fft_1d(&signal);
/// assert_eq!(spectrum.numel(), 4);
/// ```
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

/// Apollo-backed 1-D inverse FFT, reconstructing a real-valued tensor.
///
/// # Arguments
/// * `spectrum` — 1-D complex tensor of shape `[N]`.
///
/// # Returns
/// Real tensor of shape `[N]`; `ifft_1d(fft_1d(x)) == x` up to floating-point rounding.
///
/// # Panics
/// If `spectrum` is not 1-D.
///
/// # Examples
/// ```
/// use coeus_fft::{fft_1d, ifft_1d};
/// use coeus_tensor::Tensor;
///
/// let signal = Tensor::<f64>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
/// let reconstructed = ifft_1d(&fft_1d(&signal));
/// for (r, x) in reconstructed.as_slice().iter().zip([1.0, 2.0, 3.0, 4.0]) {
///     assert!((r - x).abs() < 1e-12);
/// }
/// ```
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

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;

    /// Tolerance for `N = 4`, f64 radix FFT: worst-case error grows as
    /// `O(log2(N) * eps_f64) = 2 * 2.22e-16 ~= 4.4e-16`; `1e-12` is a safe
    /// analytic margin (>3 orders of magnitude) that still rejects real defects.
    const TOL: f64 = 1e-12;

    fn assert_complex_close(got: Complex<f64>, want: Complex<f64>, ctx: &str) {
        assert!(
            (got.re - want.re).abs() < TOL && (got.im - want.im).abs() < TOL,
            "{ctx}: got {got:?}, want {want:?}"
        );
    }

    #[test]
    fn fft_matches_analytic_dft() {
        // x = [1,2,3,4]. Closed-form DFT (X[k] = sum_n x[n] exp(-2i*pi*k*n/4)):
        //   X0 = 10, X1 = -2 + 2i, X2 = -2, X3 = -2 - 2i.
        let signal = Tensor::<f64, MoiraiBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let spectrum = fft_1d(&signal);
        let out = spectrum.to_contiguous();
        let vals = out.as_slice();
        assert_complex_close(vals[0], Complex::new(10.0, 0.0), "X0");
        assert_complex_close(vals[1], Complex::new(-2.0, 2.0), "X1");
        assert_complex_close(vals[2], Complex::new(-2.0, 0.0), "X2");
        assert_complex_close(vals[3], Complex::new(-2.0, -2.0), "X3");
    }

    #[test]
    fn constant_signal_has_dc_only_spectrum() {
        // A constant signal c*[1,1,1,1] concentrates all energy at DC: X = [4c, 0, 0, 0].
        let c = 2.5;
        let signal = Tensor::<f64, MoiraiBackend>::from_slice([4], &[c, c, c, c]);
        let vals_t = fft_1d(&signal).to_contiguous();
        let vals = vals_t.as_slice();
        assert_complex_close(vals[0], Complex::new(4.0 * c, 0.0), "DC");
        for (k, v) in vals.iter().enumerate().skip(1) {
            assert_complex_close(*v, Complex::new(0.0, 0.0), &format!("bin {k}"));
        }
    }

    #[test]
    fn ifft_inverts_fft() {
        // Round-trip identity verifies Apollo's 1/N inverse normalization.
        let data = [0.5, -1.25, 3.0, 2.0, -0.75, 4.5, 1.0, -2.5];
        let signal = Tensor::<f64, MoiraiBackend>::from_slice([8], &data);
        let recon = ifft_1d(&fft_1d(&signal));
        let recon_c = recon.to_contiguous();
        for (r, x) in recon_c.as_slice().iter().zip(data) {
            assert!((r - x).abs() < TOL, "round-trip: got {r}, want {x}");
        }
    }

    #[test]
    fn f32_fft_matches_analytic_dft() {
        // Same closed form in single precision; tolerance scaled to f32 epsilon
        // (eps_f32 ~= 1.19e-7, margin to 1e-4).
        let signal = Tensor::<f32, MoiraiBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let out = fft_1d(&signal).to_contiguous();
        let vals = out.as_slice();
        let tol = 1e-4_f32;
        let want = [
            Complex::new(10.0_f32, 0.0),
            Complex::new(-2.0, 2.0),
            Complex::new(-2.0, 0.0),
            Complex::new(-2.0, -2.0),
        ];
        for (k, (g, w)) in vals.iter().zip(want).enumerate() {
            assert!(
                (g.re - w.re).abs() < tol && (g.im - w.im).abs() < tol,
                "f32 X{k}: got {g:?}, want {w:?}"
            );
        }
    }
}
