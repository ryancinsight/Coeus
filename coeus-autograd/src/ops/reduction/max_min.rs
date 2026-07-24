// ── Autograd nodes: axis reductions (max_axis, min_axis) ──
//
// # max_axis / min_axis
//   Forward: standard max/min reduction, output axis dim = 1.
//   Backward: indicator gradient — 1 at max/min positions, 0 elsewhere.
//   Ties: gradient is split equally (sum-normalised mask), matching PyTorch's
//   `max` backward convention.
//   Mask construction:
//     diff = abs(x - broadcast(max, input_shape))
//     mask_raw = ReluGrad(eps - diff)   →  1 where diff < eps (exact max match)
//     tie_count = sum_axis(mask_raw, axis)
//     mask = mask_raw / broadcast(tie_count, input_shape)
//     grad_in = broadcast(grad_out, input_shape) * mask
//   eps = T::from_f64(1e-30) — exploits the fact that floating-point max is exact.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{FloatOps, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

// ── MaxAxisNode ────────────────────────────────────────────────────────────

/// Bespoke autograd node for `max_axis`.
pub struct MaxAxisNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// Copy of the input tensor stored for backward mask construction.
    pub input_tensor: Tensor<T, B>,
    /// Forward output (max values along axis), shape has axis dim = 1.
    pub max_tensor: Tensor<T, B>,
    pub axis: usize,
}

impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for MaxAxisNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "max_axis"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else {
            return;
        };

        let max_broad = self.max_tensor.broadcast(self.input_tensor.shape_cloned());

        let eps = Tensor::full_on(self.input_tensor.shape(), T::from_f64(1e-30), &backend);
        let diff = coeus_ops::sub(&self.input_tensor, &max_broad, &backend);
        let abs_diff = coeus_ops::abs(&diff, &backend);
        let shifted = coeus_ops::sub(&eps, &abs_diff, &backend);
        let mask_raw =
            coeus_ops::elementwise_unary(&shifted, &backend, coeus_ops::UnaryOp::ReluGrad).expect("elementwise_unary");

        let tie_count = coeus_ops::sum_axis(&mask_raw, self.axis, &backend);
        let tie_broad = tie_count.broadcast(self.input_tensor.shape_cloned());
        let mask = coeus_ops::div(&mask_raw, &tie_broad, &backend);

        let grad_broad = grad_out.broadcast(self.input_tensor.shape_cloned());
        let grad_in = coeus_ops::mul(&grad_broad, &mask, &backend);

        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_in, &backend);
    }
}

/// Tracked `max_axis`: maximum over `axis`, output has that axis = 1.
///
/// Backward: indicator gradient distributed equally across tied maxima.
#[inline]
pub fn max_axis<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = coeus_ops::max_axis(&a.tensor, axis, &backend);

    let requires_grad = crate::grad_mode::should_track_var(a);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )))
    });

    let creator = requires_grad.then(|| {
        let output_grad = grad.as_ref().unwrap().clone();
        Arc::new(MaxAxisNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            max_tensor: out_tensor.clone(),
            axis,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

// ── MinAxisNode ────────────────────────────────────────────────────────────

/// Bespoke autograd node for `min_axis`.
pub struct MinAxisNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub input_tensor: Tensor<T, B>,
    pub min_tensor: Tensor<T, B>,
    pub axis: usize,
}

impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for MinAxisNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "min_axis"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else {
            return;
        };

        let min_broad = self.min_tensor.broadcast(self.input_tensor.shape_cloned());
        let eps = Tensor::full_on(self.input_tensor.shape(), T::from_f64(1e-30), &backend);
        let diff = coeus_ops::sub(&self.input_tensor, &min_broad, &backend);
        let abs_diff = coeus_ops::abs(&diff, &backend);
        let shifted = coeus_ops::sub(&eps, &abs_diff, &backend);
        let mask_raw =
            coeus_ops::elementwise_unary(&shifted, &backend, coeus_ops::UnaryOp::ReluGrad).expect("elementwise_unary");
        let tie_count = coeus_ops::sum_axis(&mask_raw, self.axis, &backend);
        let tie_broad = tie_count.broadcast(self.input_tensor.shape_cloned());
        let mask = coeus_ops::div(&mask_raw, &tie_broad, &backend);
        let grad_broad = grad_out.broadcast(self.input_tensor.shape_cloned());
        let grad_in = coeus_ops::mul(&grad_broad, &mask, &backend);

        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_in, &backend);
    }
}

/// Tracked `min_axis`: minimum over `axis`, output has that axis = 1.
///
/// Backward: indicator gradient distributed equally across tied minima.
#[inline]
pub fn min_axis<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = coeus_ops::min_axis(&a.tensor, axis, &backend);

    let requires_grad = crate::grad_mode::should_track_var(a);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )))
    });

    let creator = requires_grad.then(|| {
        let output_grad = grad.as_ref().unwrap().clone();
        Arc::new(MinAxisNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            min_tensor: out_tensor.clone(),
            axis,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
