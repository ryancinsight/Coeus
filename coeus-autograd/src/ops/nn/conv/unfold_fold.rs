// ── Tracked unfold (im2col) op ──
//
// unfold1d(input, kernel, stride, padding, dilation):
//   [N, C, L] → [N, C*kernel, L_out], extracting sliding windows.
//
// Backward:
//   d_input = fold1d(grad_out, output_size = L, kernel, stride, padding, dilation)
//   i.e. col2im — the exact transpose of im2col (each input position accumulates
//   the gradient of every window it participated in).

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

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
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
            );
            coeus_ops::add_assign(g.write(), &d_input, &backend);
        }
    }
}

/// Tracked 1D unfold (im2col): extracts sliding windows from `[N, C, L]` into
/// `[N, C*kernel_size, L_out]`, differentiable through `input` (backward is the
/// `fold1d` col2im transpose).
#[must_use]
#[inline]
pub fn unfold1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = coeus_ops::unfold1d(
        &input.tensor,
        kernel_size,
        stride,
        padding,
        dilation,
        &backend,
    );

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
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
