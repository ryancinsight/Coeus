// ── Tracked prod ──
//
// Dedicated node rather than a cumprod+slice composition: cumprod's backward
// takes the `suffix / x[i]` shortcut and emits 0 at zero-valued positions,
// but the true product gradient there is `∏_{j≠i} x_j` (generally non-zero
// when exactly one element is zero). The exact backward is selected from the
// provider-resident zero count and product:
//
//   zero-free:  grad_x[i] = grad_out · product / x[i]
//   one zero:   grad_x[i] = grad_out · product(nonzero elements) · [x[i] = 0]
//   many zero:  grad_x[i] = 0
//
// The branch avoids `0 / 0` and remains exact for any number of zeros while
// keeping the input-sized computation on the selected provider.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct ProdNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// Input tensor saved for backward.
    pub input_saved: Tensor<T, B>,
    /// Provider-resident scalar product saved for backward.
    pub product_saved: Tensor<T, B>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for ProdNode<T, B> {
    fn op_name(&self) -> &'static str {
        "prod"
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
            let seed = read_scalar(grad_out, &backend);
            let product = read_scalar(&self.product_saved, &backend);
            let zeros = Tensor::zeros_on(self.input_saved.shape_cloned(), &backend);
            let zero_mask = coeus_ops::eq(&self.input_saved, &zeros, &backend);
            let zero_count = coeus_ops::sum(&zero_mask, &backend)?;

            let gradient = if zero_count == T::zero() {
                let product_values =
                    Tensor::full_on(self.input_saved.shape_cloned(), product, &backend);
                coeus_ops::div(&product_values, &self.input_saved, &backend)
            } else if zero_count == T::one() {
                let nonzero_input = coeus_ops::add(&self.input_saved, &zero_mask, &backend);
                let nonzero_product = coeus_ops::prod(&nonzero_input, &backend);
                let nonzero_values =
                    Tensor::full_on(self.input_saved.shape_cloned(), nonzero_product, &backend);
                coeus_ops::mul(&nonzero_values, &zero_mask, &backend)
            } else {
                Tensor::zeros_on(self.input_saved.shape_cloned(), &backend)
            };

            let scaled_gradient = if seed == T::one() {
                gradient
            } else {
                let seed_values = Tensor::full_on(self.input_saved.shape_cloned(), seed, &backend);
                coeus_ops::mul(&gradient, &seed_values, &backend)
            };
            let gl = g.write();
            coeus_ops::add_assign(gl, &scaled_gradient, &backend)?;
        }
        Ok(())
    }
}

fn read_scalar<T: Scalar, B: ComputeBackend>(tensor: &Tensor<T, B>, backend: &B) -> T {
    let mut scalar = [T::zero()];
    backend.copy_to_host(tensor.storage(), &mut scalar);
    scalar[0]
}

/// Tracked product of all elements (`torch.prod`), returning a `[1]` tensor.
///
/// Backward: `d prod/dx_i = ∏_{j≠i} x_j`, computed exactly with provider
/// equality, reduction, fill, multiplication, and division operations. This
/// is valid for zero and negative elements (unlike `exp(sum(log x))` or a
/// cumprod-based composition).
///
/// # Panics
/// Panics if `input` is empty.
#[must_use]
#[inline]
pub fn prod<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    assert!(
        input.tensor.numel() > 0,
        "prod: empty tensors have no product"
    );
    let backend = B::default();
    let out_tensor =
        coeus_ops::prod_tensor(&input.tensor, &backend).expect("prod: provider reduction failed");

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
        let node = ProdNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            input_saved: input.tensor.clone(),
            product_saved: out_tensor.clone(),
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
