//! Reverse-mode autograd nodes for the Apollo-backed 1-D FFT.

use crate::fft::{accumulate_grad, fft_1d, ifft_1d, tensor_to_vec, FftScalar};
use coeus_autograd::BackwardNode;
use coeus_autograd::GradBuffer;
use coeus_autograd::Var;
use coeus_core::{Complex, ComputeBackend, MoiraiBackend};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Backward node for `fft_1d_var`.
pub struct Fft1DNode<T: FftScalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Real input variable.
    pub x: Var<T, B>,
    /// Output gradient buffer in the complex frequency domain.
    pub output_grad: Arc<GradBuffer<Complex<T>, B>>,
}

impl<T, B> BackwardNode<Complex<T>, B> for Fft1DNode<T, B>
where
    T: FftScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "fft_1d"
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
        let n = T::from_usize(grad_out.numel());
        let mut dx = ifft_1d(grad_out);
        let mut host = tensor_to_vec(&dx);
        for value in &mut host {
            *value *= n;
        }
        dx = Tensor::from_slice_on(dx.shape_cloned(), &host, &B::default());

        if let Some(ref grad) = self.x.grad {
            accumulate_grad(grad, &dx);
        }
        if self.x.creator.is_some() {
            if let Some(current_grad) = self.x.grad() {
                self.x.backward_with_seed(current_grad)?;
            }
        }
        Ok(())
    }
}

/// Backward node for `ifft_1d_var`.
pub struct Ifft1DNode<T: FftScalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Complex spectrum variable.
    pub y: Var<Complex<T>, B>,
    /// Output gradient buffer in the real domain.
    pub output_grad: Arc<GradBuffer<T, B>>,
}

impl<T, B> BackwardNode<T, B> for Ifft1DNode<T, B>
where
    T: FftScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "ifft_1d"
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
        let n = T::from_usize(grad_out.numel());
        let mut dy = fft_1d(grad_out);
        let mut host = tensor_to_vec(&dy);
        for value in &mut host {
            value.re = value.re / n;
            value.im = value.im / n;
        }
        dy = Tensor::from_slice_on(dy.shape_cloned(), &host, &B::default());

        if let Some(ref grad) = self.y.grad {
            accumulate_grad(grad, &dy);
        }
        if self.y.creator.is_some() {
            if let Some(current_grad) = self.y.grad() {
                self.y.backward_with_seed(current_grad)?;
            }
        }
        Ok(())
    }
}

/// Differentiable Apollo-backed 1-D forward FFT.
#[must_use]
pub fn fft_1d_var<T, B>(x: &Var<T, B>) -> Var<Complex<T>, B>
where
    T: FftScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let out_tensor = fft_1d(&x.tensor);
    let requires_grad = coeus_autograd::is_grad_enabled() && x.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        Arc::new(Fft1DNode {
            x: x.clone(),
            output_grad,
        }) as Arc<dyn BackwardNode<Complex<T>, B>>
    });

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

/// Differentiable Apollo-backed 1-D inverse FFT.
#[must_use]
pub fn ifft_1d_var<T, B>(y: &Var<Complex<T>, B>) -> Var<T, B>
where
    T: FftScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let out_tensor = ifft_1d(&y.tensor);
    let requires_grad = coeus_autograd::is_grad_enabled() && y.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        Arc::new(Ifft1DNode {
            y: y.clone(),
            output_grad,
        }) as Arc<dyn BackwardNode<T, B>>
    });

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

struct FftEnergyNode<T: FftScalar, B: ComputeBackend + Default = MoiraiBackend> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    spectrum: Tensor<Complex<T>, B>,
}

impl<T, B> BackwardNode<T, B> for FftEnergyNode<T, B>
where
    T: FftScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "fft_energy"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        if let Some(Some(ref grad)) = input_grads.first() {
            let go = tensor_to_vec(grad_out)[0];
            let factor = go * T::from_f64(2.0);
            let spec_host = tensor_to_vec(&self.spectrum);
            let grad_spec: Vec<Complex<T>> = spec_host
                .iter()
                .map(|c| Complex::new(c.re * factor, c.im * factor))
                .collect();
            let grad_spec =
                Tensor::from_slice_on(self.spectrum.shape_cloned(), &grad_spec, &B::default());
            let mut dx = ifft_1d(&grad_spec);
            let n = T::from_usize(dx.numel());
            let mut dx_host = tensor_to_vec(&dx);
            for value in &mut dx_host {
                *value *= n;
            }
            dx = Tensor::from_slice_on(dx.shape_cloned(), &dx_host, &B::default());
            accumulate_grad(grad, &dx);
        }
        Ok(())
    }
}

/// Sum of squared FFT magnitudes, preserving gradients to the real input.
#[must_use]
pub fn fft_energy<T, B>(x: &Var<T, B>) -> Var<T, B>
where
    T: FftScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let spectrum = fft_1d(&x.tensor);
    let energy = tensor_to_vec(&spectrum)
        .iter()
        .fold(T::zero(), |acc, c| acc + c.re * c.re + c.im * c.im);
    let out_tensor = Tensor::from_slice_on([1], &[energy], &backend);
    let requires_grad = coeus_autograd::is_grad_enabled() && x.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        Arc::new(FftEnergyNode {
            output_grad,
            inputs: vec![x.clone()],
            spectrum,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
