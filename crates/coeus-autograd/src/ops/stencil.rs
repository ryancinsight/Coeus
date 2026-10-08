//! Tracked stencil operators: the staggered gradient/divergence pair.
//!
//! The pair is a negative adjoint (`D = -Gᵀ`), so each half's backward pass
//! is the other half applied to the upstream gradient and accumulated with a
//! negation — no new kernels, on any backend that implements the pair.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_ops::{Axis, BackendOps, ElementwiseOps, StaggeredPairOps};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd graph connections and the retained prepared pair.
pub struct StaggeredGradNode<T, B>
where
    T: Scalar,
    B: BackendOps<T> + StaggeredPairOps<T> + ElementwiseOps<T> + Default,
{
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: [Var<T, B>; 1],
    pair: B::StaggeredPair,
    axis: Axis,
}

/// Autograd graph connections and the retained prepared pair.
pub struct StaggeredDivNode<T, B>
where
    T: Scalar,
    B: BackendOps<T> + StaggeredPairOps<T> + ElementwiseOps<T> + Default,
{
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: [Var<T, B>; 1],
    pair: B::StaggeredPair,
    axis: Axis,
}

fn adjoint_into<T, B>(
    pair: &B::StaggeredPair,
    axis: Axis,
    grad_out: &Tensor<T, B>,
    gradient: &mut Tensor<T, B>,
    divergence: bool,
) -> Result<(), B::Error>
where
    T: Float,
    B: BackendOps<T> + StaggeredPairOps<T> + ElementwiseOps<T> + Default,
{
    let backend = B::default();
    let mut adjoint = Tensor::zeros_on(grad_out.shape().to_vec(), &backend);
    let (storage, layout) = adjoint.storage_mut_and_layout();
    if divergence {
        backend.staggered_divergence(
            pair,
            axis,
            grad_out.storage(),
            grad_out.layout(),
            storage,
            layout,
        )?;
    } else {
        backend.staggered_gradient(
            pair,
            axis,
            grad_out.storage(),
            grad_out.layout(),
            storage,
            layout,
        )?;
    }
    coeus_ops::sub_assign(gradient, &adjoint, &backend)
}

impl<T, B> BackwardNode<T, B> for StaggeredGradNode<T, B>
where
    T: Float,
    B: BackendOps<T> + StaggeredPairOps<T> + ElementwiseOps<T> + Default,
{
    fn op_name(&self) -> &'static str {
        "staggered_gradient"
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
        if let Some(Some(gradient)) = input_grads.first() {
            adjoint_into::<T, B>(&self.pair, self.axis, grad_out, gradient.write(), true)?;
        }
        Ok(())
    }
}

impl<T, B> BackwardNode<T, B> for StaggeredDivNode<T, B>
where
    T: Float,
    B: BackendOps<T> + StaggeredPairOps<T> + ElementwiseOps<T> + Default,
{
    fn op_name(&self) -> &'static str {
        "staggered_divergence"
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
        if let Some(Some(gradient)) = input_grads.first() {
            adjoint_into::<T, B>(&self.pair, self.axis, grad_out, gradient.write(), false)?;
        }
        Ok(())
    }
}

fn tracked_pair<T, B>(
    input: &Var<T, B>,
    order: usize,
    spacing: [T; 3],
    axis: Axis,
    divergence: bool,
) -> Result<Var<T, B>, B::Error>
where
    T: Float,
    B: BackendOps<T> + StaggeredPairOps<T> + ElementwiseOps<T> + Default,
{
    let backend = B::default();
    let pair = backend.prepare_staggered_pair(order, spacing)?;
    let mut output = Tensor::zeros_on(input.tensor.shape().to_vec(), &backend);
    let (storage, layout) = output.storage_mut_and_layout();
    if divergence {
        backend.staggered_divergence(
            &pair,
            axis,
            input.tensor.storage(),
            input.tensor.layout(),
            storage,
            layout,
        )?;
    } else {
        backend.staggered_gradient(
            &pair,
            axis,
            input.tensor.storage(),
            input.tensor.layout(),
            storage,
            layout,
        )?;
    }
    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            input.tensor.shape().to_vec(),
            &backend,
        )))
    });
    let creator = grad.as_ref().map(|output_grad| {
        // The existing graph erases heterogeneous operation nodes at its graph boundary.
        if divergence {
            Arc::new(StaggeredDivNode {
                output_grad: Arc::clone(output_grad),
                inputs: [input.clone()],
                pair,
                axis,
            }) as Arc<dyn BackwardNode<T, B>>
        } else {
            Arc::new(StaggeredGradNode {
                output_grad: Arc::clone(output_grad),
                inputs: [input.clone()],
                pair,
                axis,
            }) as Arc<dyn BackwardNode<T, B>>
        }
    });
    Ok(Var {
        tensor: output,
        grad,
        creator,
    })
}

/// Tracked staggered gradient along `axis` at even `order`.
///
/// The input is a rank-3 field; preparation derives the stencil taps once and
/// the node retains the pair for backward, where the gradient's adjoint is
/// the negated divergence. Runs on any backend implementing the pair — CPU
/// over Leto or a GPU over Hephaestus.
///
/// # Errors
///
/// Returns the backend's typed error for an invalid order, spacing, or field
/// layout, or the provider's dispatch failure.
pub fn staggered_gradient<T, B>(
    input: &Var<T, B>,
    order: usize,
    spacing: [T; 3],
    axis: Axis,
) -> Result<Var<T, B>, B::Error>
where
    T: Float,
    B: BackendOps<T> + StaggeredPairOps<T> + ElementwiseOps<T> + Default,
{
    tracked_pair(input, order, spacing, axis, false)
}

/// Tracked staggered divergence along `axis` at even `order`.
///
/// The `-Gᵀ` half of the pair; backward applies the negated gradient. See
/// [`staggered_gradient`].
///
/// # Errors
///
/// See [`staggered_gradient`].
pub fn staggered_divergence<T, B>(
    input: &Var<T, B>,
    order: usize,
    spacing: [T; 3],
    axis: Axis,
) -> Result<Var<T, B>, B::Error>
where
    T: Float,
    B: BackendOps<T> + StaggeredPairOps<T> + ElementwiseOps<T> + Default,
{
    tracked_pair(input, order, spacing, axis, true)
}
