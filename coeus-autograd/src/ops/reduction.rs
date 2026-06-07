// ── Autograd nodes: axis reductions (max_axis, min_axis, log_sum_exp) ──
//
// `cumsum` is implemented in shape.rs (tracked, backward = suffix_sum).
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
//
// # log_sum_exp
//   Composed entirely from existing tracked ops — no new backward node needed.
//     lse(x, axis) = log(sum(exp(x − max_axis(x)), axis)) + max_axis(x)
//   max-subtraction stabilises the exp range to (−∞, 0] preventing overflow.
//   Gradient flows through max_axis → sub → exp → sum_axis → log → add.

use std::sync::{Arc, Mutex};
use coeus_core::{Scalar, Float};
use coeus_tensor::Tensor;
use crate::node::BackwardNode;
use crate::var::Var;
use crate::ops::arithmetic::{add, sub, sum_axis};
use crate::ops::activation::{exp, log};

// ── MaxAxisNode ────────────────────────────────────────────────────────────

/// Bespoke autograd node for `max_axis`.
pub struct MaxAxisNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad:  Arc<Mutex<Tensor<T, B>>>,
    pub inputs:       Vec<Var<T, B>>,
    /// Copy of the input tensor stored for backward mask construction.
    pub input_tensor: Tensor<T, B>,
    /// Forward output (max values along axis), shape has axis dim = 1.
    pub max_tensor:   Tensor<T, B>,
    pub axis:         usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for MaxAxisNode<T, B> {
    #[inline] fn op_name(&self) -> &'static str { "max_axis" }
    #[inline] fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> { &self.output_grad }
    #[inline] fn inputs(&self) -> &[Var<T, B>] { &self.inputs }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else { return };

        // 1. Broadcast max_tensor (axis dim = 1) back to full input shape.
        let max_broad = self.max_tensor.broadcast(self.input_tensor.shape_cloned());

        // 2. Build indicator mask: 1 at max positions, 0 elsewhere.
        //    diff = abs(x - max_broad)  →  0 exactly at max positions.
        //    ReluGrad(eps - diff): 1 where diff < eps (i.e. at exact max), 0 otherwise.
        let eps = Tensor::full_on(self.input_tensor.shape(), T::from_f64(1e-30), &backend);
        let diff      = coeus_ops::sub(&self.input_tensor, &max_broad, &backend);
        let abs_diff  = coeus_ops::abs(&diff, &backend);
        let shifted   = coeus_ops::sub(&eps, &abs_diff, &backend);
        let mask_raw  = coeus_ops::elementwise_unary(&shifted, &backend, coeus_ops::UnaryOp::ReluGrad);

        // 3. Tie-normalise: divide mask by the count of max-valued elements along axis.
        let tie_count = coeus_ops::sum_axis(&mask_raw, self.axis, &backend);
        let tie_broad = tie_count.broadcast(self.input_tensor.shape_cloned());
        let mask = coeus_ops::div(&mask_raw, &tie_broad, &backend);

        // 4. Broadcast grad_out to input shape and apply mask.
        let grad_broad = grad_out.broadcast(self.input_tensor.shape_cloned());
        let grad_in = coeus_ops::mul(&grad_broad, &mask, &backend);

        let mut lock = g.lock().unwrap();
        coeus_ops::add_assign(&mut *lock, &grad_in, &backend);
    }
}

/// Tracked `max_axis`: maximum over `axis`, output has that axis = 1.
///
/// Backward: indicator gradient distributed equally across tied maxima.
#[inline]
pub fn max_axis<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = coeus_ops::max_axis(&a.tensor, axis, &backend);

    let requires_grad = a.grad.is_some();
    let grad = requires_grad
        .then(|| Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))));

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
    Var { tensor: out_tensor, grad, creator }
}

// ── MinAxisNode ────────────────────────────────────────────────────────────

/// Bespoke autograd node for `min_axis`.
pub struct MinAxisNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad:  Arc<Mutex<Tensor<T, B>>>,
    pub inputs:       Vec<Var<T, B>>,
    pub input_tensor: Tensor<T, B>,
    pub min_tensor:   Tensor<T, B>,
    pub axis:         usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for MinAxisNode<T, B> {
    #[inline] fn op_name(&self) -> &'static str { "min_axis" }
    #[inline] fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> { &self.output_grad }
    #[inline] fn inputs(&self) -> &[Var<T, B>] { &self.inputs }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else { return };

        // Identical logic to MaxAxisNode but using min_tensor.
        let min_broad = self.min_tensor.broadcast(self.input_tensor.shape_cloned());
        let eps       = Tensor::full_on(self.input_tensor.shape(), T::from_f64(1e-30), &backend);
        let diff      = coeus_ops::sub(&self.input_tensor, &min_broad, &backend);
        let abs_diff  = coeus_ops::abs(&diff, &backend);
        let shifted   = coeus_ops::sub(&eps, &abs_diff, &backend);
        let mask_raw  = coeus_ops::elementwise_unary(&shifted, &backend, coeus_ops::UnaryOp::ReluGrad);
        let tie_count = coeus_ops::sum_axis(&mask_raw, self.axis, &backend);
        let tie_broad = tie_count.broadcast(self.input_tensor.shape_cloned());
        let mask      = coeus_ops::div(&mask_raw, &tie_broad, &backend);
        let grad_broad = grad_out.broadcast(self.input_tensor.shape_cloned());
        let grad_in   = coeus_ops::mul(&grad_broad, &mask, &backend);

        let mut lock = g.lock().unwrap();
        coeus_ops::add_assign(&mut *lock, &grad_in, &backend);
    }
}

/// Tracked `min_axis`: minimum over `axis`, output has that axis = 1.
///
/// Backward: indicator gradient distributed equally across tied minima.
#[inline]
pub fn min_axis<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = coeus_ops::min_axis(&a.tensor, axis, &backend);

    let requires_grad = a.grad.is_some();
    let grad = requires_grad
        .then(|| Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))));

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
    Var { tensor: out_tensor, grad, creator }
}

// ── log_sum_exp ────────────────────────────────────────────────────────────
//
// Numerically stable log-sum-exp along `axis`.
//
// Composed from existing tracked ops so that grad flows through the DAG
// automatically without a bespoke node:
//
//   x_max  = max_axis(x, axis)             → tracked, broadcasts back as size-1 dim
//   x_sh   = x - x_max                     → stable shifted input, exp in (-inf, 0]
//   sum_e  = sum_axis(exp(x_sh), axis)      → sum of stabilised exponentials
//   lse    = log(sum_e) + x_max             → add back the max offset
//
// The backward through log and exp reproduces the softmax probabilities, so
// d(lse)/dx_i = softmax(x)_i — correct by composition.

/// Tracked numerically stable log-sum-exp along `axis`.
///
/// Output shape equals input shape with `axis` dimension reduced to 1.
/// No new backward node — gradients flow through the composed tracked ops.
///
/// Precision: all computation in native `T` precision; max-subtraction
/// constrains exp input to (−∞, 0], eliminating overflow at any precision.
#[inline]
pub fn log_sum_exp<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    let x_max     = max_axis(x, axis);          // shape: [..., 1, ...]
    let x_shifted = sub(x, &x_max);             // broadcasts axis=1 dim over x
    let exp_sh    = exp(&x_shifted);
    let sum_exp   = sum_axis(&exp_sh, axis);
    let log_sum   = log(&sum_exp);
    add(&log_sum, &x_max)                        // adds back max offset
}
