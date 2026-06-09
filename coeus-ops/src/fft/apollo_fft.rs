// ── Apollo FFT integration ──
// 1-D FFT/IFFT using the Apollo library via its slice/Vec API (no ndarray here;
// coeus's array stack is self-contained — ndarray lives only inside apollo).

use apollo_fft::{Complex32, Complex64};
use coeus_core::{Complex, ComputeBackend, Float, Storage};
use coeus_tensor::Tensor;

/// Sealed trait for compile-time monomorphized FFT dispatch.
pub trait FftScalar: Float {
    /// Internal Complex type used by Apollo FFT.
    type Complex;

    /// Direct 1-D forward FFT implementation.
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>>;

    /// Direct 1-D inverse FFT implementation.
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self>;
}

impl FftScalar for f64 {
    type Complex = Complex64;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        apollo_fft::fft_1d_slice_typed::<f64>(signal)
            .into_iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect()
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let spec: Vec<Complex64> = spectrum
            .iter()
            .map(|c| Complex64::new(c.re, c.im))
            .collect();
        apollo_fft::ifft_1d_slice_typed::<f64>(&spec)
    }
}

impl FftScalar for f32 {
    type Complex = Complex32;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        apollo_fft::fft_1d_slice_typed::<f32>(signal)
            .into_iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect()
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let spec: Vec<Complex32> = spectrum
            .iter()
            .map(|c| Complex32::new(c.re, c.im))
            .collect();
        apollo_fft::ifft_1d_slice_typed::<f32>(&spec)
    }
}

impl FftScalar for half::f16 {
    type Complex = Complex32;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        apollo_fft::fft_1d_slice_typed::<half::f16>(signal)
            .into_iter()
            .map(|c| Complex::new(half::f16::from_f32(c.re), half::f16::from_f32(c.im)))
            .collect()
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let spec: Vec<Complex32> = spectrum
            .iter()
            .map(|c| Complex32::new(c.re.to_f32(), c.im.to_f32()))
            .collect();
        apollo_fft::ifft_1d_slice_typed::<half::f16>(&spec)
    }
}

impl FftScalar for half::bf16 {
    type Complex = Complex32;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        let input: Vec<f32> = signal.iter().map(|&x| x.to_f32()).collect();
        apollo_fft::fft_1d_slice_typed::<f32>(&input)
            .into_iter()
            .map(|c| Complex::new(half::bf16::from_f32(c.re), half::bf16::from_f32(c.im)))
            .collect()
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let spec: Vec<Complex32> = spectrum
            .iter()
            .map(|c| Complex32::new(c.re.to_f32(), c.im.to_f32()))
            .collect();
        apollo_fft::ifft_1d_slice_typed::<f32>(&spec)
            .into_iter()
            .map(half::bf16::from_f32)
            .collect()
    }
}

/// 1-D forward FFT. Returns a Complex tensor.
#[inline]
pub fn fft_1d<T: FftScalar, B: ComputeBackend + Default>(
    signal: &Tensor<T, B>,
) -> Tensor<Complex<T>, B> {
    assert_eq!(signal.ndim(), 1, "fft_1d requires 1D input");
    let input = signal.to_contiguous();
    let numel = input.numel();
    let out_vec = if let Some(slice) = input.storage().try_as_slice() {
        T::fft_1d_impl(slice)
    } else {
        let mut host_buf = vec![T::zero(); numel];
        B::default().copy_to_host(input.storage(), &mut host_buf);
        T::fft_1d_impl(&host_buf)
    };
    let mut out = Tensor::zeros_on([numel], &B::default());
    B::default().copy_to_device(&out_vec, out.storage_mut());
    out
}

/// 1-D inverse FFT from Complex component.
#[inline]
pub fn ifft_1d<T: FftScalar, B: ComputeBackend + Default>(
    spectrum: &Tensor<Complex<T>, B>,
) -> Tensor<T, B> {
    assert_eq!(spectrum.ndim(), 1, "ifft_1d requires 1D input");
    let spectrum_cont = spectrum.to_contiguous();
    let numel = spectrum_cont.numel();
    let out_vec = if let Some(slice) = spectrum_cont.storage().try_as_slice() {
        T::ifft_1d_impl(slice)
    } else {
        let mut host_buf = vec![Complex::new(T::zero(), T::zero()); numel];
        B::default().copy_to_host(spectrum_cont.storage(), &mut host_buf);
        T::ifft_1d_impl(&host_buf)
    };
    let mut out = Tensor::zeros_on([numel], &B::default());
    B::default().copy_to_device(&out_vec, out.storage_mut());
    out
}
