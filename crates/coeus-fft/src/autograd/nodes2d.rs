//! Reverse-mode autograd nodes for the Apollo-backed 2-D FFT.

use crate::fft::{accumulate_grad, tensor_to_vec};
use crate::fft2d::{fft_2d, ifft_2d, Fft2DScalar};
use coeus_autograd::BackwardNode;
use coeus_autograd::GradBuffer;
use coeus_autograd::Var;
use coeus_core::{Complex, ComputeBackend, MoiraiBackend};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Differentiable Apollo-backed 2-D forward FFT.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn fft_2d_var<T, B>(x: &Var<T, B>) -> Result<Var<Complex<T>, B>, B::Error>
where
    T: Fft2DScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let out_tensor = fft_2d(&x.tensor)?;
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
        Arc::new(Fft2DNode {
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

/// Backward node for `fft_2d_var`.
pub struct Fft2DNode<T: Fft2DScalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Real input variable.
    pub x: Var<T, B>,
    /// Output gradient buffer in the complex frequency domain.
    pub output_grad: Arc<GradBuffer<Complex<T>, B>>,
}

impl<T, B> BackwardNode<Complex<T>, B> for Fft2DNode<T, B>
where
    T: Fft2DScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "fft_2d"
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
        let n = T::from_count(grad_out.numel());
        let mut dx = ifft_2d(grad_out)?;
        let mut host = tensor_to_vec(&dx)?;
        for value in &mut host {
            *value *= n;
        }
        dx = Tensor::from_slice_on(dx.shape_cloned(), &host, &B::default())?;

        if let Some(ref grad) = self.x.grad {
            accumulate_grad(grad, &dx)?;
        }
        if self.x.creator.is_some() {
            if let Some(current_grad) = self.x.grad() {
                self.x.backward_with_seed(current_grad)?;
            }
        }
        Ok(())
    }
}

/// Backward node for `ifft_2d_var`.
pub struct Ifft2DNode<T: Fft2DScalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Complex spectrum variable.
    pub y: Var<Complex<T>, B>,
    /// Output gradient buffer in the real domain.
    pub output_grad: Arc<GradBuffer<T, B>>,
}

impl<T, B> BackwardNode<T, B> for Ifft2DNode<T, B>
where
    T: Fft2DScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "ifft_2d"
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
        let n = T::from_count(grad_out.numel());
        let mut dy = fft_2d(grad_out)?;
        let mut host = tensor_to_vec(&dy)?;
        for value in &mut host {
            value.re = value.re / n;
            value.im = value.im / n;
        }
        dy = Tensor::from_slice_on(dy.shape_cloned(), &host, &B::default())?;

        if let Some(ref grad) = self.y.grad {
            accumulate_grad(grad, &dy)?;
        }
        if self.y.creator.is_some() {
            if let Some(current_grad) = self.y.grad() {
                self.y.backward_with_seed(current_grad)?;
            }
        }
        Ok(())
    }
}

/// Differentiable Apollo-backed 2-D inverse FFT.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn ifft_2d_var<T, B>(y: &Var<Complex<T>, B>) -> Result<Var<T, B>, B::Error>
where
    T: Fft2DScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let out_tensor = ifft_2d(&y.tensor)?;
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
        Arc::new(Ifft2DNode {
            y: y.clone(),
            output_grad,
        }) as Arc<dyn BackwardNode<T, B>>
    });

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
