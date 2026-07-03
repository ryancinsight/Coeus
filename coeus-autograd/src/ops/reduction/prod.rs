// ── Tracked prod ──
//
// Dedicated node rather than a cumprod+slice composition: cumprod's backward
// takes the `suffix / x[i]` shortcut and emits 0 at zero-valued positions,
// but the true product gradient there is `∏_{j≠i} x_j` (generally non-zero
// when exactly one element is zero). The exact backward is computed with
// prefix/suffix products in O(n) and native `T` precision:
//
//   grad_x[i] = grad_out · prefix[i] · suffix[i]
//   prefix[i] = ∏_{j<i} x_j,  suffix[i] = ∏_{j>i} x_j
//
// which is exact for any number of zeros (two or more zeros ⇒ all gradients
// zero, one zero ⇒ only the zero position has non-zero gradient).

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct ProdNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// Input tensor saved for backward.
    pub input_saved: Tensor<T, B>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for ProdNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "prod"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let go = grad_out.to_contiguous();
            let seed = go.as_slice()[0];
            let in_cont = self.input_saved.to_contiguous();
            let xs = in_cont.as_slice();
            let n = xs.len();

            // prefix[i] = ∏_{j<i} x_j accumulated forward; suffix accumulated
            // in a reverse sweep, multiplied in place: gi[i] = prefix[i]·suffix[i].
            let mut gi = vec![T::one(); n];
            let mut acc = T::one();
            for i in 0..n {
                gi[i] = acc;
                acc = acc * xs[i];
            }
            let mut acc = T::one();
            for i in (0..n).rev() {
                gi[i] = seed * gi[i] * acc;
                acc = acc * xs[i];
            }

            let gi_increment = Tensor::from_slice(self.input_saved.shape_cloned(), &gi);
            let gl = g.write();
            coeus_ops::add_assign(gl, &gi_increment, &backend);
        }
    }
}

/// Tracked product of all elements (`torch.prod`), returning a `[1]` tensor.
///
/// Backward: `d prod/dx_i = ∏_{j≠i} x_j`, computed exactly via prefix/suffix
/// products — valid for zero and negative elements (unlike `exp(sum(log x))`
/// or a cumprod-based composition).
///
/// # Panics
/// Panics if `input` is empty.
#[must_use]
#[inline]
pub fn prod<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    assert!(
        input.tensor.numel() > 0,
        "prod: empty tensors have no product"
    );
    let backend = B::default();
    let value = coeus_ops::prod(&input.tensor, &backend);
    let out_tensor = Tensor::from_slice_on(vec![1], &[value], &backend);

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
