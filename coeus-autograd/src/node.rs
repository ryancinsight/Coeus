// ── Computation graph node ──

use crate::grad_buffer::GradBuffer;
use crate::var::Var;
use coeus_core::{ComputeBackend, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// A node in the autograd DAG, recording an operation for backward propagation.
///
/// Each differentiable op produces a node that stores its input [`Var`]s and
/// saved forward tensors, exposes its output gradient buffer, and implements
/// [`BackwardNode::backward`] to push gradients into its inputs given the
/// upstream output gradient. [`Var::backward`](crate::Var::backward) traverses
/// the DAG in reverse topological order, calling each node's `backward`.
///
/// # Examples
///
/// A minimal node for `y = c * x` (a scaled leaf), showing the four required
/// methods. The node holds the constant `c`, accumulates the output gradient,
/// and pushes `c * grad_out` into the input's gradient buffer using
/// [`GradBuffer::write`](crate::GradBuffer::write).
///
/// ```
/// use coeus_autograd::{BackwardNode, GradBuffer, Var};
/// use coeus_core::{ComputeBackend, MoiraiBackend, Scalar};
/// use coeus_tensor::Tensor;
/// use std::sync::Arc;
///
/// struct ScaleNode<T: Scalar, B: ComputeBackend + Default = MoiraiBackend> {
///     output_grad: Arc<GradBuffer<T, B>>,
///     inputs: Vec<Var<T, B>>,
///     c: T,
/// }
///
/// impl<T: Scalar, B: ComputeBackend + Default> BackwardNode<T, B>
///     for ScaleNode<T, B>
/// where
///     B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
/// {
///     fn op_name(&self) -> &'static str { "scale" }
///     fn output_grad(&self) -> &Arc<GradBuffer<T, B>> { &self.output_grad }
///     fn inputs(&self) -> &[Var<T, B>] { &self.inputs }
///     fn backward(
///         &self,
///         grad_out: &Tensor<T, B>,
///         input_grads: &[Option<Arc<GradBuffer<T, B>>>],
///     ) {
///         // d/dx (c * x) = c, so grad_x += c * grad_out
///         if let Some(Some(ref g)) = input_grads.get(0) {
///             let go = grad_out.as_slice();
///             let gx = g.write();
///             let gx_s = gx.as_mut_slice();
///             for i in 0..gx_s.len() {
///                 // Real accumulation: grad_x[i] += c * grad_out[i]
///                 let c_val = T::from_f64(<T as Scalar>::to_f64(self.c));
///                 let go_val = T::from_f64(<T as Scalar>::to_f64(go[i]));
///                 let acc = T::from_f64(<T as Scalar>::to_f64(gx_s[i]) + <T as Scalar>::to_f64(c_val) * <T as Scalar>::to_f64(go_val));
///                 gx_s[i] = acc;
///             }
///         }
///     }
/// }
/// ```
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
