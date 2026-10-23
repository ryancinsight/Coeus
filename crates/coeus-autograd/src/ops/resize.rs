//! Differentiable spatial resize (nearest / bilinear / bicubic / area).
//!
//! Reverse mode for [`coeus_ops::resize`]: the forward pass and the backward
//! scatter share one tap table, so the gradient is exactly the transpose of the
//! linear (or piecewise-constant) map the forward pass applies.

use crate::{grad_buffer::GradBuffer, node::BackwardNode, var::Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement};
use coeus_ops::ResizeMode;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Reverse-mode node for a 1-D resize: shape and mode are all the reverse pass
/// needs, since the map is linear in the input.
struct Resize1dNode<T, B>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
{
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    input_shape: Vec<usize>,
    new_l: usize,
    /// Sampling step in source units per output position, carried so the
    /// scatter uses exactly the taps the forward pass applied.
    scale: f64,
    mode: ResizeMode,
}

impl<T, B> BackwardNode<T, B> for Resize1dNode<T, B>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "resize_1d"
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
        let gradient = coeus_ops::resize_1d_backward_with_scale(
            &self.input_shape,
            self.new_l,
            self.scale,
            self.mode,
            grad_out,
        )?;
        let backend = B::default();
        if let Some(Some(accumulator)) = input_grads.first() {
            coeus_ops::add_assign(accumulator.write(), &gradient, &backend)?;
        }
        Ok(())
    }
}

/// Reverse-mode node for a 2-D resize.
struct Resize2dNode<T, B>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
{
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    input_shape: Vec<usize>,
    new_h: usize,
    new_w: usize,
    scale_h: f64,
    scale_w: f64,
    mode: ResizeMode,
}

impl<T, B> BackwardNode<T, B> for Resize2dNode<T, B>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "resize_2d"
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
        let gradient = coeus_ops::resize_2d_backward_with_scale(
            &self.input_shape,
            self.new_h,
            self.new_w,
            self.scale_h,
            self.scale_w,
            self.mode,
            grad_out,
        )?;
        let backend = B::default();
        if let Some(Some(accumulator)) = input_grads.first() {
            coeus_ops::add_assign(accumulator.write(), &gradient, &backend)?;
        }
        Ok(())
    }
}

/// Resize a tracked `[N, C, L]` tensor to `[N, C, new_l]`.
///
/// Nearest-neighbour scatters the upstream gradient onto the source positions
/// it copied; bilinear applies the transposed blend, so the two source taps of
/// each output position each receive their weight's share.
pub fn resize_1d<T, B>(
    input: &Var<T, B>,
    new_l: usize,
    mode: ResizeMode,
) -> Result<Var<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let l = input.tensor.shape()[2];
    resize_1d_with_scale(input, new_l, l as f64 / new_l as f64, mode)
}

/// [`resize_1d`] with an explicit sampling step of `scale` source units per
/// output position — pass `1 / scale_factor` for a `scale_factor` resize, since
/// torch samples at the requested factor rather than at the rounded output
/// size.
pub fn resize_1d_with_scale<T, B>(
    input: &Var<T, B>,
    new_l: usize,
    scale: f64,
    mode: ResizeMode,
) -> Result<Var<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let output = coeus_ops::resize_1d_with_scale(&input.tensor, new_l, scale, mode)?;
    let requires_grad = crate::grad_mode::should_track_var(input);
    if !requires_grad {
        return Var::new(output, false);
    }

    let backend = B::default();
    let input_shape = input.tensor.shape().to_vec();
    Var::from_tracked_op(output, requires_grad, &backend, |output_grad| {
        Resize1dNode::<T, B> {
            output_grad,
            inputs: vec![input.clone()],
            input_shape,
            new_l,
            scale,
            mode,
        }
    })
}

/// Resize a tracked `[N, C, H, W]` tensor to `[N, C, new_h, new_w]`.
///
/// See [`resize_1d`] for the gradient semantics; the 2-D case is the separable
/// product of the two axis blends.
pub fn resize_2d<T, B>(
    input: &Var<T, B>,
    new_h: usize,
    new_w: usize,
    mode: ResizeMode,
) -> Result<Var<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.tensor.shape();
    resize_2d_with_scale(
        input,
        new_h,
        new_w,
        shape[2] as f64 / new_h as f64,
        shape[3] as f64 / new_w as f64,
        mode,
    )
}

/// [`resize_2d`] with explicit per-axis sampling steps — see
/// [`resize_1d_with_scale`].
pub fn resize_2d_with_scale<T, B>(
    input: &Var<T, B>,
    new_h: usize,
    new_w: usize,
    scale_h: f64,
    scale_w: f64,
    mode: ResizeMode,
) -> Result<Var<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let output =
        coeus_ops::resize_2d_with_scale(&input.tensor, new_h, new_w, scale_h, scale_w, mode)?;
    let requires_grad = crate::grad_mode::should_track_var(input);
    if !requires_grad {
        return Var::new(output, false);
    }

    let backend = B::default();
    let input_shape = input.tensor.shape().to_vec();
    Var::from_tracked_op(output, requires_grad, &backend, |output_grad| {
        Resize2dNode::<T, B> {
            output_grad,
            inputs: vec![input.clone()],
            input_shape,
            new_h,
            new_w,
            scale_h,
            scale_w,
            mode,
        }
    })
}
