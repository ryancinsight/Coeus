//! Reverse-mode autograd nodes for the multidimensional half-spectrum FFT.
//!
//! `rfft_2d_var`/`rfft_3d_var` mirror the half gradient to a full Hermitian
//! spectrum and route through the full-length adjoint (`N * ifft`), the same
//! composition as the 1-D half node. `irfft_2d_var`/`irfft_3d_var` take the
//! full forward FFT of the real upstream gradient and fold the mirror lanes
//! back into the half spectrum (doubling interior lanes, singling the
//! self-mirror corners).

use crate::fft::{accumulate_grad, tensor_to_vec};
use crate::fft2d::{fft_2d, ifft_2d};
use crate::fft3d::{fft_3d, ifft_3d};
use crate::rfftn::{
    fold_half_2d, fold_half_3d, irfft_2d, irfft_3d, mirror_complete_2d, mirror_complete_3d,
    rfft_2d, rfft_3d, FftHalfNdScalar,
};
use coeus_autograd::{BackwardNode, GradBuffer, Var};
use coeus_core::{Complex, ComputeBackend, MoiraiBackend};
use coeus_tensor::Tensor;
use std::sync::Arc;

fn mirrored_adjoint_2d<T, B>(grad_out: &Tensor<Complex<T>, B>, nx: usize, ny: usize) -> Result<Tensor<T, B>, B::Error>
where
    T: FftHalfNdScalar + crate::fft2d::Fft2DScalar,
    B: ComputeBackend + Default,
{
    let half_host = tensor_to_vec(grad_out)?;
    let full = mirror_complete_2d(&half_host, nx, ny);
    let full_tensor = Tensor::from_slice_on(vec![nx, ny], &full, &B::default())?;
    let n = T::from_count(nx * ny);
    let dx = ifft_2d(&full_tensor)?;
    let mut host = tensor_to_vec(&dx)?;
    for value in &mut host {
        *value *= n;
    }
    Tensor::from_slice_on(dx.shape_cloned(), &host, &B::default())
}

fn mirrored_adjoint_3d<T, B>(
    grad_out: &Tensor<Complex<T>, B>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<Tensor<T, B>, B::Error>
where
    T: FftHalfNdScalar + crate::fft3d::Fft3DScalar,
    B: ComputeBackend + Default,
{
    let half_host = tensor_to_vec(grad_out)?;
    let full = mirror_complete_3d(&half_host, nx, ny, nz);
    let full_tensor = Tensor::from_slice_on(vec![nx, ny, nz], &full, &B::default())?;
    let n = T::from_count(nx * ny * nz);
    let dx = ifft_3d(&full_tensor)?;
    let mut host = tensor_to_vec(&dx)?;
    for value in &mut host {
        *value *= n;
    }
    Tensor::from_slice_on(dx.shape_cloned(), &host, &B::default())
}

/// Backward node for `rfft_2d_var`.
pub struct Rfft2DNode<T: FftHalfNdScalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Real input variable.
    pub x: Var<T, B>,
    /// Output gradient buffer in the half-spectrum domain.
    pub output_grad: Arc<GradBuffer<Complex<T>, B>>,
    /// Full input width (the half shape alone is parity-ambiguous).
    pub nx: usize,
    /// Full input width (the half shape alone is parity-ambiguous).
    pub ny: usize,
}

impl<T, B> BackwardNode<Complex<T>, B> for Rfft2DNode<T, B>
where
    T: FftHalfNdScalar + crate::fft2d::Fft2DScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "rfft_2d"
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
        let dx = mirrored_adjoint_2d(grad_out, self.nx, self.ny)?;
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

/// Differentiable Apollo-backed half-spectrum 2-D forward FFT.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn rfft_2d_var<T, B>(x: &Var<T, B>) -> Result<Var<Complex<T>, B>, B::Error>
where
    T: FftHalfNdScalar + crate::fft2d::Fft2DScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let shape = x.tensor.shape();
    let (nx, ny) = (shape[0], shape[1]);
    let out_tensor = rfft_2d(&x.tensor)?;
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
        Arc::new(Rfft2DNode {
            x: x.clone(),
            output_grad,
            nx,
            ny,
        }) as Arc<dyn BackwardNode<Complex<T>, B>>
    });

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

/// Backward node for `irfft_2d_var`.
pub struct Irfft2DNode<T: FftHalfNdScalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Half-spectrum variable.
    pub y: Var<Complex<T>, B>,
    /// Output gradient buffer in the real domain.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Full output dimensions.
    pub nx: usize,
    /// Full output dimensions.
    pub ny: usize,
}

impl<T, B> BackwardNode<T, B> for Irfft2DNode<T, B>
where
    T: FftHalfNdScalar + crate::fft2d::Fft2DScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "irfft_2d"
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
        let full = fft_2d(grad_out)?;
        let full_host = tensor_to_vec(&full)?;
        let scale = T::from_count(self.nx * self.ny);
        let half = fold_half_2d(&full_host, self.nx, self.ny, scale);
        let dy = Tensor::from_slice_on(
            vec![self.nx, self.ny / 2 + 1],
            &half,
            &B::default(),
        )?;

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

/// Differentiable Apollo-backed half-spectrum 2-D inverse FFT.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn irfft_2d_var<T, B>(
    y: &Var<Complex<T>, B>,
    nx: usize,
    ny: usize,
) -> Result<Var<T, B>, B::Error>
where
    T: FftHalfNdScalar + crate::fft2d::Fft2DScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let out_tensor = irfft_2d(&y.tensor, nx, ny)?;
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
        Arc::new(Irfft2DNode {
            y: y.clone(),
            output_grad,
            nx,
            ny,
        }) as Arc<dyn BackwardNode<T, B>>
    });

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

/// Backward node for `rfft_3d_var`.
pub struct Rfft3DNode<T: FftHalfNdScalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Real input variable.
    pub x: Var<T, B>,
    /// Output gradient buffer in the half-spectrum domain.
    pub output_grad: Arc<GradBuffer<Complex<T>, B>>,
    /// Full input width (the half shape alone is parity-ambiguous).
    pub nx: usize,
    /// Full input width (the half shape alone is parity-ambiguous).
    pub ny: usize,
    /// Full input width (the half shape alone is parity-ambiguous).
    pub nz: usize,
}

impl<T, B> BackwardNode<Complex<T>, B> for Rfft3DNode<T, B>
where
    T: FftHalfNdScalar + crate::fft3d::Fft3DScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "rfft_3d"
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
        let dx = mirrored_adjoint_3d(grad_out, self.nx, self.ny, self.nz)?;
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

/// Differentiable Apollo-backed half-spectrum 3-D forward FFT.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn rfft_3d_var<T, B>(x: &Var<T, B>) -> Result<Var<Complex<T>, B>, B::Error>
where
    T: FftHalfNdScalar + crate::fft3d::Fft3DScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let shape = x.tensor.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let out_tensor = rfft_3d(&x.tensor)?;
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
        Arc::new(Rfft3DNode {
            x: x.clone(),
            output_grad,
            nx,
            ny,
            nz,
        }) as Arc<dyn BackwardNode<Complex<T>, B>>
    });

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

/// Backward node for `irfft_3d_var`.
pub struct Irfft3DNode<T: FftHalfNdScalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Half-spectrum variable.
    pub y: Var<Complex<T>, B>,
    /// Output gradient buffer in the real domain.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Full output dimensions.
    pub nx: usize,
    /// Full output dimensions.
    pub ny: usize,
    /// Full output dimensions.
    pub nz: usize,
}

impl<T, B> BackwardNode<T, B> for Irfft3DNode<T, B>
where
    T: FftHalfNdScalar + crate::fft3d::Fft3DScalar,
    B: ComputeBackend + Default,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "irfft_3d"
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
        let full = fft_3d(grad_out)?;
        let full_host = tensor_to_vec(&full)?;
        let scale = T::from_count(self.nx * self.ny * self.nz);
        let half = fold_half_3d(&full_host, self.nx, self.ny, self.nz, scale);
        let dy = Tensor::from_slice_on(
            vec![self.nx, self.ny, self.nz / 2 + 1],
            &half,
            &B::default(),
        )?;

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

/// Differentiable Apollo-backed half-spectrum 3-D inverse FFT.
///
/// # Errors
///
/// Returns the backend error when reading the input or allocating the output fails.
pub fn irfft_3d_var<T, B>(
    y: &Var<Complex<T>, B>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<Var<T, B>, B::Error>
where
    T: FftHalfNdScalar + crate::fft3d::Fft3DScalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    let out_tensor = irfft_3d(&y.tensor, nx, ny, nz)?;
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
        Arc::new(Irfft3DNode {
            y: y.clone(),
            output_grad,
            nx,
            ny,
            nz,
        }) as Arc<dyn BackwardNode<T, B>>
    });

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
