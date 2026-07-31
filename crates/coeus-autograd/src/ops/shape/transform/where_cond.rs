// ── Tracked where_cond ──
//
// Backward of where_cond(cond, on_true, on_false):
//   d on_true  += grad_out * any_mask
//   d on_false += grad_out * (1 - any_mask)
//   d cond     = 0  (indicator function; no gradient through the condition)

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct WhereCond<T: coeus_core::Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// The computed mask (`1` where cond != 0, `0` elsewhere), reused for backward.
    pub any_mask: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for WhereCond<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "where_cond"
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
        // inputs: [cond=0, on_true=1, on_false=2]
        // grad cond  = 0 (no derivative through boolean mask)
        // grad true  = grad_out * any_mask
        // grad false = grad_out * (1 - any_mask)
        if let Some(Some(ref g)) = input_grads.get(1) {
            let d_true = coeus_ops::mul(grad_out, &self.any_mask, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &d_true, &backend)?;
        }
        if let Some(Some(ref g)) = input_grads.get(2) {
            let one = Tensor::full_on(self.any_mask.shape(), T::from_f64(1.0), &backend);
            let inv = coeus_ops::sub(&one, &self.any_mask, &backend);
            let d_false = coeus_ops::mul(grad_out, &inv, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &d_false, &backend)?;
        }
        Ok(())
    }
}

/// Tracked conditional element-wise select.
///
/// Gradient flows to `on_true` and `on_false`; `cond` receives zero gradient
/// (indicator function is non-differentiable at 0).
///
/// # Panics
/// Panics if `cond`, `on_true`, and `on_false` do not have the same shape.
#[must_use]
#[inline]
pub fn where_cond<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    cond: &Var<T, B>,
    on_true: &Var<T, B>,
    on_false: &Var<T, B>,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();

    // Compute mask once; reuse in backward.
    let mask_pos =
        coeus_ops::elementwise_unary(&cond.tensor, &backend, coeus_ops::UnaryOp::ReluGrad)
            .expect("elementwise_unary");
    let cond_neg = coeus_ops::elementwise_unary(&cond.tensor, &backend, coeus_ops::UnaryOp::Neg)
        .expect("elementwise_unary");
    let mask_neg = coeus_ops::elementwise_unary(&cond_neg, &backend, coeus_ops::UnaryOp::ReluGrad)
        .expect("elementwise_unary");
    let any_mask = coeus_ops::add(&mask_pos, &mask_neg, &backend);

    let one = Tensor::full_on(any_mask.shape(), T::from_f64(1.0), &backend);
    let inv_mask = coeus_ops::sub(&one, &any_mask, &backend);
    let true_part = coeus_ops::mul(&on_true.tensor, &any_mask, &backend);
    let false_part = coeus_ops::mul(&on_false.tensor, &inv_mask, &backend);
    let out_tensor = coeus_ops::add(&true_part, &false_part, &backend);

    let requires_grad = crate::grad_mode::should_track_var(cond)
        || crate::grad_mode::should_track_var(on_true)
        || crate::grad_mode::should_track_var(on_false);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let node = WhereCond {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![cond.clone(), on_true.clone(), on_false.clone()],
            any_mask,
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
