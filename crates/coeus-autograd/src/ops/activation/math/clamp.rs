//! Tracked element-wise clamp.
//!
//! Gradient of clamp is 1 inside [min, max] and 0 outside.  This is the
//! straight-through estimator for the STE variant; for exact grad it is the
//! indicator function.  Here we implement the indicator (standard definition):
//!
//!   d/dx clamp(x, lo, hi) = 1_{lo ≤ x ≤ hi}
//!
//! which means: gradient passes through unchanged where the input is inside the
//! clamp bounds, and is zeroed where x < lo or x > hi.
//!
//! Implementation via two ReluGrad applications on shifted inputs (zero-copy
//! composition of existing backend primitives, no new kernel needed):
//!   inside = ReluGrad(x - lo) * ReluGrad(hi - x)

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{FloatOps, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for element-wise clamp.
pub struct ClampNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Input tensor snapshot for backward mask computation.
    pub input_tensor: Tensor<T, B>,
    /// Lower bound of the clamp.
    pub min_val: T,
    /// Upper bound of the clamp.
    pub max_val: T,
}

impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for ClampNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "clamp"
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
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let shape = self.input_tensor.shape();
            let one_scalar = Tensor::full_on(shape, T::ONE, &backend);

            // 1_{lo <= x} = 1 - 1_{x < lo} = 1 - ReluGrad(lo - x).
            // At the kink x = lo: (lo - x) = 0, ReluGrad(0) = 0 (strict),
            // so the indicator evaluates to 1 — matching PyTorch's
            // `aten::clamp_backward_kernel` which is `[min, max]` inclusive.
            let lo_t = Tensor::full_on(shape, self.min_val, &backend);
            let lo_minus_x = coeus_ops::sub(&lo_t, &self.input_tensor, &backend);
            let mask_lt_lo =
                coeus_ops::elementwise_unary(&lo_minus_x, &backend, coeus_ops::UnaryOp::ReluGrad)?;
            let mask_lo_ge = coeus_ops::sub(&one_scalar, &mask_lt_lo, &backend);

            // 1_{x <= hi} = 1 - 1_{x > hi} = 1 - ReluGrad(x - hi).
            // At the kink x = hi: (x - hi) = 0, ReluGrad(0) = 0 (strict),
            // so the upper indicator evaluates to 1 — same inclusive
            // convention as the lower bound.
            let hi_t = Tensor::full_on(shape, self.max_val, &backend);
            let x_minus_hi = coeus_ops::sub(&self.input_tensor, &hi_t, &backend);
            let mask_gt_hi =
                coeus_ops::elementwise_unary(&x_minus_hi, &backend, coeus_ops::UnaryOp::ReluGrad)?;
            let mask_hi_le = coeus_ops::sub(&one_scalar, &mask_gt_hi, &backend);

            // Inside mask = mask_lo_ge AND mask_hi_le (product of indicators)
            let inside = coeus_ops::mul(&mask_lo_ge, &mask_hi_le, &backend);
            let grad_in = coeus_ops::mul(grad_out, &inside, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &grad_in, &backend)?;
        }
        Ok(())
    }
}

/// Tracked element-wise clamp: `y = clamp(x, min_val, max_val)`.
///
/// Gradient is the indicator `1_{min_val ≤ x ≤ max_val}` — passes through
/// where input is in range, zeroed at saturated positions.
#[must_use]
#[inline]
pub fn clamp<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    min_val: T,
    max_val: T,
) -> Var<T, B> {
    let backend = B::default();

    // Forward: clamp element-wise.
    // Implemented via existing primitives: clamp(x, lo, hi) = min(max(x, lo), hi)
    // = relu_shift approach using two relu operations on shifted values.
    // More precisely: clip from below then from above.
    let lo_t = Tensor::full_on(a.tensor.shape(), min_val, &backend);
    let hi_t = Tensor::full_on(a.tensor.shape(), max_val, &backend);
    // x_clamped_lo = relu(x - lo) + lo = max(x, lo)
    let shifted_lo = coeus_ops::sub(&a.tensor, &lo_t, &backend);
    let relu_lo = coeus_ops::elementwise_unary(&shifted_lo, &backend, coeus_ops::UnaryOp::Relu)
        .expect("elementwise_unary");
    let clamped_lo = coeus_ops::add(&relu_lo, &lo_t, &backend);
    // max(x, lo) clamped from above: min(clamped_lo, hi) = hi - relu(hi - clamped_lo)
    let shifted_hi = coeus_ops::sub(&hi_t, &clamped_lo, &backend);
    let relu_hi = coeus_ops::elementwise_unary(&shifted_hi, &backend, coeus_ops::UnaryOp::Relu)
        .expect("elementwise_unary");
    let out_tensor = coeus_ops::sub(&hi_t, &relu_hi, &backend);

    let requires_grad = crate::grad_mode::should_track_var(a);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().expect("invariant: requires_grad gates both the Some(grad) construction above and this read").clone();
        let node = ClampNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            min_val,
            max_val,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
