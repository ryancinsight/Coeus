// ── Computation graph node ──

use crate::grad_buffer::GradBuffer;
use crate::var::Var;
use coeus_core::{ComputeBackend, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// A node in the autograd DAG, recording an operation for backward propagation.
pub trait BackwardNode<T: Scalar, B: ComputeBackend + Default = MoiraiBackend>:
    Send + Sync
{
    /// Human-readable operation name.
    fn op_name(&self) -> &'static str;

    /// Accumulated gradient for this node's output.
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>>;

    /// Input variables that produced this node's output.
    fn inputs(&self) -> &[Var<T, B>];

    /// Backward function: given the output gradient, push gradients to inputs.
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]);
}
