// ── Apollo FFT integration ──
// 1-D FFT/IFFT using the Apollo library.

use coeus_core::{Float, Complex, ComputeBackend, Storage};
use coeus_tensor::Tensor;
use apollo_fft::{Complex32, Complex64};

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
        let n = signal.len();
        let cast_slice: &[f64] = bytemuck::cast_slice(signal);
        let arr = ndarray::ArrayView1::from_shape(n, cast_slice).unwrap();
        let owned_arr = arr.to_owned();
        let out_arr = apollo_fft::fft_1d_array_typed::<f64>(&owned_arr);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(Complex::new(out_arr[i].re, out_arr[i].im));
        }
        out
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let n = spectrum.len();
        let mut complex_vec = Vec::with_capacity(n);
        for i in 0..n {
            complex_vec.push(Complex64::new(spectrum[i].re, spectrum[i].im));
        }
        let arr = ndarray::Array1::from_vec(complex_vec);
        let out_arr = apollo_fft::ifft_1d_array_typed::<f64>(&arr);
        out_arr.to_vec()
    }
}

impl FftScalar for f32 {
    type Complex = Complex32;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        let n = signal.len();
        let cast_slice: &[f32] = bytemuck::cast_slice(signal);
        let arr = ndarray::ArrayView1::from_shape(n, cast_slice).unwrap();
        let owned_arr = arr.to_owned();
        let out_arr = apollo_fft::fft_1d_array_typed::<f32>(&owned_arr);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(Complex::new(out_arr[i].re, out_arr[i].im));
        }
        out
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let n = spectrum.len();
        let mut complex_vec = Vec::with_capacity(n);
        for i in 0..n {
            complex_vec.push(Complex32::new(spectrum[i].re, spectrum[i].im));
        }
        let arr = ndarray::Array1::from_vec(complex_vec);
        let out_arr = apollo_fft::ifft_1d_array_typed::<f32>(&arr);
        out_arr.to_vec()
    }
}

impl FftScalar for half::f16 {
    type Complex = Complex32;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        let n = signal.len();
        let cast_slice: &[half::f16] = bytemuck::cast_slice(signal);
        let arr = ndarray::ArrayView1::from_shape(n, cast_slice).unwrap();
        let owned_arr = arr.to_owned();
        let out_arr = apollo_fft::fft_1d_array_typed::<half::f16>(&owned_arr);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(Complex::new(
                half::f16::from_f32(out_arr[i].re),
                half::f16::from_f32(out_arr[i].im),
            ));
        }
        out
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let n = spectrum.len();
        let mut complex_vec = Vec::with_capacity(n);
        for i in 0..n {
            complex_vec.push(Complex32::new(spectrum[i].re.to_f32(), spectrum[i].im.to_f32()));
        }
        let arr = ndarray::Array1::from_vec(complex_vec);
        let out_arr = apollo_fft::ifft_1d_array_typed::<half::f16>(&arr);
        out_arr.to_vec()
    }
}

impl FftScalar for half::bf16 {
    type Complex = Complex32;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        let n = signal.len();
        let input_vec: Vec<f32> = signal.iter().map(|&x| x.to_f32()).collect();
        let arr = ndarray::Array1::from_vec(input_vec);
        let out_arr = apollo_fft::fft_1d_array_typed::<f32>(&arr);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(Complex::new(
                half::bf16::from_f32(out_arr[i].re),
                half::bf16::from_f32(out_arr[i].im),
            ));
        }
        out
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let n = spectrum.len();
        let mut complex_vec = Vec::with_capacity(n);
        for i in 0..n {
            complex_vec.push(Complex32::new(spectrum[i].re.to_f32(), spectrum[i].im.to_f32()));
        }
        let arr = ndarray::Array1::from_vec(complex_vec);
        let out_arr = apollo_fft::ifft_1d_array_typed::<f32>(&arr);
        out_arr.iter().map(|&x| half::bf16::from_f32(x)).collect()
    }
}

/// 1-D forward FFT. Returns a Complex tensor.
#[inline]
pub fn fft_1d<T: FftScalar, B: ComputeBackend + Default>(signal: &Tensor<T, B>) -> Tensor<Complex<T>, B> {
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
pub fn ifft_1d<T: FftScalar, B: ComputeBackend + Default>(spectrum: &Tensor<Complex<T>, B>) -> Tensor<T, B> {
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
