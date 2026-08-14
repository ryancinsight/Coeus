// ── Tracked unfold (im2col) op ──
//
// unfold1d(input, kernel, stride, padding, dilation):
//   [N, C, L] → [N, C*kernel, L_out], extracting sliding windows.
//
// Backward:
//   d_input = fold1d(grad_out, output_size = L, kernel, stride, padding, dilation)
//   i.e. col2im — the exact transpose of im2col (each input position accumulates
//   the gradient of every window it participated in).
#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

struct Unfold1dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    /// Original input length `L`, used as `fold1d`'s `output_size` in backward.
    output_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for Unfold1dNode<T, B> {
    fn op_name(&self) -> &'static str {
        "unfold1d"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            // col2im: fold the windowed gradient back onto the input positions.
            let d_input = coeus_ops::fold1d(
                grad_out,
                self.output_size,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                &backend,
            )?;
            coeus_ops::add_assign(g.write(), &d_input, &backend)?;
        }

        Ok(())
    }
}

/// Tracked 1D unfold (im2col): extracts sliding windows from `[N, C, L]` into
/// `[N, C*kernel_size, L_out]`, differentiable through `input` (backward is the
/// `fold1d` col2im transpose).
///
/// # Errors
///
/// Returns the backend error when unfold validation or dispatch fails.
#[inline]
pub fn unfold1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let out_tensor = coeus_ops::unfold1d(
        &input.tensor,
        kernel_size,
        stride,
        padding,
        dilation,
        &backend,
    )?;

    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let node = Unfold1dNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            output_size: input.tensor.shape()[2],
            kernel_size,
            stride,
            padding,
            dilation,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

// ── unfold2d (im2col), backward = fold2d (col2im) ──

struct Unfold2dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for Unfold2dNode<T, B> {
    fn op_name(&self) -> &'static str {
        "unfold2d"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let d_input = coeus_ops::fold2d(
                grad_out,
                self.output_h,
                self.output_w,
                self.kernel_h,
                self.kernel_w,
                self.stride_h,
                self.stride_w,
                self.padding_h,
                self.padding_w,
                self.dilation_h,
                self.dilation_w,
                &backend,
            )?;
            coeus_ops::add_assign(g.write(), &d_input, &backend)?;
        }

        Ok(())
    }
}

/// Tracked 2D unfold (im2col) over `[N, C, H, W]`, differentiable through
/// `input` (backward is the `fold2d` col2im transpose).
///
/// # Errors
///
/// Returns the backend error when unfold validation or dispatch fails.
#[inline]
#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub fn unfold2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let out_tensor = coeus_ops::unfold2d(
        &input.tensor,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        &backend,
    )?;
    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )))
    });
    let creator = grad.as_ref().map(|grad| {
        let shape = input.tensor.shape();
        Arc::new(Unfold2dNode {
            output_grad: grad.clone(),
            inputs: vec![input.clone()],
            output_h: shape[2],
            output_w: shape[3],
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

// ── fold1d (col2im), backward = unfold1d (im2col) ──

struct Fold1dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for Fold1dNode<T, B> {
    fn op_name(&self) -> &'static str {
        "fold1d"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            // Adjoint of col2im is im2col over the output-shaped gradient.
            let d_input = coeus_ops::unfold1d(
                grad_out,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                &backend,
            )?;
            coeus_ops::add_assign(g.write(), &d_input, &backend)?;
        }

        Ok(())
    }
}

/// Tracked 1D fold (col2im): accumulates `[N, C*kernel, L_out]` back into
/// `[N, C, output_size]`, differentiable through `input` (backward is `unfold1d`).
///
/// # Errors
///
/// Returns the backend error when fold validation or dispatch fails.
#[inline]
pub fn fold1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    output_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let out_tensor = coeus_ops::fold1d(
        &input.tensor,
        output_size,
        kernel_size,
        stride,
        padding,
        dilation,
        &backend,
    )?;
    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )))
    });
    let creator = grad.as_ref().map(|grad| {
        Arc::new(Fold1dNode {
            output_grad: grad.clone(),
            inputs: vec![input.clone()],
            kernel_size,
            stride,
            padding,
            dilation,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

// ── fold2d (col2im), backward = unfold2d (im2col) ──

struct Fold2dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for Fold2dNode<T, B> {
    fn op_name(&self) -> &'static str {
        "fold2d"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let d_input = coeus_ops::unfold2d(
                grad_out,
                self.kernel_h,
                self.kernel_w,
                self.stride_h,
                self.stride_w,
                self.padding_h,
                self.padding_w,
                self.dilation_h,
                self.dilation_w,
                &backend,
            )?;
            coeus_ops::add_assign(g.write(), &d_input, &backend)?;
        }

        Ok(())
    }
}

/// Tracked 2D fold (col2im) into `[N, C, output_h, output_w]`, differentiable
/// through `input` (backward is the `unfold2d` im2col adjoint).
///
/// # Errors
///
/// Returns the backend error when fold validation or dispatch fails.
#[inline]
#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub fn fold2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let out_tensor = coeus_ops::fold2d(
        &input.tensor,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        &backend,
    )?;
    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )))
    });
    let creator = grad.as_ref().map(|grad| {
        Arc::new(Fold2dNode {
            output_grad: grad.clone(),
            inputs: vec![input.clone()],
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
