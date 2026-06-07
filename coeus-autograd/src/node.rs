// ── Computation graph node ──

use std::sync::{Arc, Mutex};
use coeus_core::{Scalar, ComputeBackend, MoiraiBackend};
use coeus_tensor::Tensor;
use crate::var::Var;

/// A node in the autograd DAG, recording an operation for backward propagation.
pub trait BackwardNode<T: Scalar, B: ComputeBackend + Default = MoiraiBackend>: Send + Sync {
    /// Human-readable operation name.
    fn op_name(&self) -> &'static str;

    /// Accumulated gradient for this node's output.
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>>;

    /// Input variables that produced this node's output.
    fn inputs(&self) -> &[Var<T, B>];

    /// Backward function: given the output gradient, push gradients to inputs.
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]);
}

