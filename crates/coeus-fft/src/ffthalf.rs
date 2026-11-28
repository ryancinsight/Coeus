//! Apollo-backed half-spectrum 1-D FFT kernels for Coeus tensors.
//!
//! A real signal's spectrum satisfies `X[n-k] = conj(X[k])`, so the upper
//! half carries no information. [`rfft_1d`] returns only the `n/2 + 1`
//! independent bins via [`apollo_fft::fft_1d_slice_half`]; [`irfft_1d`]
//! inverts a half spectrum via [`apollo_fft::ifft_1d_slice_half_into`].
//! `irfft_1d(rfft_1d(x)) == x` up to floating-point rounding.
//!
//! The tracked wrappers compose the existing full-length adjoint pair: the
//! backward mirrors the half gradient to a full spectrum (conjugate upper
//! half) and routes through the full-length [`crate::fft`] transforms, so
//! no new adjoint identity is introduced. Placement matches the other FFT
//! wrappers: host-side Apollo transforms, no new `BackendOp` opcode.

use coeus_autograd::{BackwardNode, GradBuffer, Var};
use coeus_core::{Complex, ComputeBackend, Float, MoiraiBackend};
use coeus_tensor::Tensor;
use std::ops::Neg;
use std::sync::Arc;

/// Scalar types supported by the half-spectrum transforms.
///
/// `f32`/`f64` only, matching [`crate::fft::FftScalar`], which this trait
/// extends: both backward nodes compose the full-length transforms, so
/// every half-scalar is a full scalar.
pub trait FftHalfScalar: crate::fft::FftScalar {
    /// Compute the `n/2 + 1` independent bins of a real signal's forward FFT.
    fn rfft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>>;

    /// Invert an `n/2 + 1`-bin half spectrum into `n` reals.
    fn irfft_1d_impl(spectrum: &mut [Complex<Self>], n: usize) -> Vec<Self>;
}

impl FftHalfScalar for f32 {
    #[inline]
    fn rfft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        apollo_fft::fft_1d_slice_half(signal)
    }

    #[inline]
    fn irfft_1d_impl(spectrum: &mut [Complex<Self>], n: usize) -> Vec<Self> {
        let mut out = vec![0.0; n];
        apollo_fft::ifft_1d_slice_half_into(spectrum, &mut out);
        out
    }
}

impl FftHalfScalar for f64 {
    #[inline]
    fn rfft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        apollo_fft::fft_1d_slice_half(signal)
    }

    #[inline]
    fn irfft_1d_impl(spectrum: &mut [Complex<Self>], n: usize) -> Vec<Self> {
        let mut out = vec![0.0; n];
        apollo_fft::ifft_1d_slice_half_into(spectrum, &mut out);
        out
    }
}

/// Mirror a half spectrum to its full-length Hermitian completion.
///
/// Bins `1..n/2` reflect as conjugates; the DC and (even-`n`) Nyquist bins
/// are their own images. Panics when `half.len() != n / 2 + 1`.
fn mirror_to_full<T: Float + Neg<Output = T> + Copy>(
    half: &[Complex<T>],
    n: usize,
) -> Vec<Complex<T>> {
    assert_eq!(
        half.len(),
        n / 2 + 1,
        "mirror_to_full: half spectrum must hold exactly n/2 + 1 bins"
    );
    let mut full = Vec::with_capacity(n);
    full.extend_from_slice(half);
    full.extend(
        half[1..=n - half.len()]
            .iter()
            .rev()
            .map(|bin| Complex::new(bin.re, -bin.im)),
    );
    full
}

/// Apollo-backed half-spectrum forward FFT for Coeus tensors.
///
/// The input must be 1-D with length `n`; the output holds `n/2 + 1`
/// complex bins. Unnormalized.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn rfft_1d<T, B>(signal: &Tensor<T, B>) -> Result<Tensor<Complex<T>, B>, B::Error>
where
    T: FftHalfScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(signal.ndim(), 1, "rfft_1d requires 1-D input");
    let backend = B::default();
    let input = crate::fft::tensor_to_vec(signal)?;
    let spectrum = T::rfft_1d_impl(&input);
    Tensor::from_slice_on(vec![spectrum.len()], &spectrum, &backend)
}

/// Apollo-backed half-spectrum inverse FFT for Coeus tensors.
///
/// The input must be a 1-D half spectrum of length `n/2 + 1`; the output is
/// `n` reals. Divides by `n`.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn irfft_1d<T, B>(spectrum: &Tensor<Complex<T>, B>, n: usize) -> Result<Tensor<T, B>, B::Error>
where
    T: FftHalfScalar,
    B: ComputeBackend + Default,
{
    assert_eq!(spectrum.ndim(), 1, "irfft_1d requires 1-D input");
    assert!(
        spectrum.numel() == n / 2 + 1,
        "irfft_1d: half spectrum must hold exactly n/2 + 1 bins"
    );
    let backend = B::default();
    let mut input = crate::fft::tensor_to_vec(spectrum)?;
    let signal = T::irfft_1d_impl(&mut input, n);
    Tensor::from_slice_on(vec![n], &signal, &backend)
}

/// Backward node for `rfft_1d_var`.
pub struct Rfft1DNode<T: FftHalfScalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Real input variable.
    pub x: Var<T, B>,
    /// Output gradient buffer in the half-spectrum domain.
    pub output_grad: Arc<GradBuffer<Complex<T>, B>>,
}

impl<T, B> BackwardNode<Complex<T>, B> for Rfft1DNode<T, B>
where
    T: FftHalfScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "rfft_1d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<Complex<T>, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<Complex<T>, B>] {
        &[]
    }

    fn backward(
        &self,
        grad_out: &Tensor<Complex<T>, B>,
        _input_grads: &[Option<Arc<GradBuffer<Complex<T>, B>>>],
    ) -> Result<(), B::Error> {
        let n = grad_out.numel() * 2 - 2;
        let half_host = crate::fft::tensor_to_vec(grad_out)?;
        let full = mirror_to_full(&half_host, n);
        let full_tensor = Tensor::from_slice_on(vec![n], &full, &B::default())?;
        let scale = T::from_count(n);
        let mut dx = crate::fft::ifft_1d(&full_tensor)?;
        let mut host = crate::fft::tensor_to_vec(&dx)?;
        for value in &mut host {
            *value *= scale;
        }
        dx = Tensor::from_slice_on(dx.shape_cloned(), &host, &B::default())?;

        if let Some(ref grad) = self.x.grad {
            crate::fft::accumulate_grad(grad, &dx)?;
        }
        if self.x.creator.is_some() {
            if let Some(current_grad) = self.x.grad() {
                self.x.backward_with_seed(current_grad)?;
            }
        }
        Ok(())
    }
}

/// Differentiable Apollo-backed half-spectrum forward FFT.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn rfft_1d_var<T, B>(x: &Var<T, B>) -> Result<Var<Complex<T>, B>, B::Error>
where
    T: FftHalfScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let out_tensor = rfft_1d(&x.tensor)?;
    let requires_grad = coeus_autograd::is_grad_enabled() && x.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        Arc::new(Rfft1DNode {
            x: x.clone(),
            output_grad,
        }) as Arc<dyn BackwardNode<Complex<T>, B>>
    });

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

/// Backward node for `irfft_1d_var`.
pub struct Irfft1DNode<T: FftHalfScalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Half-spectrum variable.
    pub y: Var<Complex<T>, B>,
    /// Output gradient buffer in the real domain.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Output length the forward produced.
    pub n: usize,
}

impl<T, B> BackwardNode<T, B> for Irfft1DNode<T, B>
where
    T: FftHalfScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "irfft_1d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &[]
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        _input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        // Adjoint of (mirror completion, then forward FFT): the completion
        // fans each interior half bin out to two full lanes (k and N-k as
        // conjugates), so its adjoint folds them back in. DC and (even-n)
        // Nyquist bins are their own images and fold singly.
        let n = self.n;
        let full = crate::fft::fft_1d(grad_out)?;
        let full_host = crate::fft::tensor_to_vec(&full)?;
        let scale = T::from_count(n);
        let half_len = n / 2 + 1;
        let mut half = Vec::with_capacity(half_len);
        for (k, value) in full_host.iter().take(half_len).enumerate() {
            let folded = if k == 0 || (n.is_multiple_of(2) && k * 2 == n) {
                *value
            } else {
                let mirror = full_host[n - k];
                Complex::new(value.re + mirror.re, value.im - mirror.im)
            };
            half.push(Complex::new(folded.re / scale, folded.im / scale));
        }
        let dy = Tensor::from_slice_on(vec![half_len], &half, &B::default())?;

        if let Some(ref grad) = self.y.grad {
            crate::fft::accumulate_grad(grad, &dy)?;
        }
        if self.y.creator.is_some() {
            if let Some(current_grad) = self.y.grad() {
                self.y.backward_with_seed(current_grad)?;
            }
        }
        Ok(())
    }
}

/// Differentiable Apollo-backed half-spectrum inverse FFT.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn irfft_1d_var<T, B>(y: &Var<Complex<T>, B>, n: usize) -> Result<Var<T, B>, B::Error>
where
    T: FftHalfScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let out_tensor = irfft_1d(&y.tensor, n)?;
    let requires_grad = coeus_autograd::is_grad_enabled() && y.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        Arc::new(Irfft1DNode {
            y: y.clone(),
            output_grad,
            n,
        }) as Arc<dyn BackwardNode<T, B>>
    });

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
