//! Linear voxel-coordinate interpolation with tracked image and grid gradients.

use crate::{grad_buffer::GradBuffer, node::BackwardNode, var::Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement};
use coeus_ops::{
    linear_interpolation_backward, BoundaryPolicy, Dimension, InterpolationFailure, Replicate,
    SupportedDimension,
};
use coeus_tensor::Tensor;
use std::{marker::PhantomData, sync::Arc};

/// Reverse-mode node for linear sampling.
pub struct LinearInterpolationNode<T, const D: usize, B, P = Replicate>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
    P: BoundaryPolicy,
{
    /// Accumulated output gradient.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Image and sampling-grid variables.
    pub inputs: Vec<Var<T, B>>,
    /// Saved image values required by the coordinate derivative.
    pub image: Tensor<T, B>,
    /// Saved sampling coordinates required by both derivatives.
    pub grid: Tensor<T, B>,
    policy: PhantomData<P>,
}

impl<T, const D: usize, B, P> BackwardNode<T, B> for LinearInterpolationNode<T, D, B, P>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
    P: BoundaryPolicy + Send + Sync + 'static,
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "linear_interpolation"
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
        let updates = linear_interpolation_backward::<T, D, _, P>(
            &self.image,
            &self.grid,
            grad_out,
            P::default(),
        )
        .map_err(|error| {
            B::Error::from(coeus_core::BackendError::Storage {
                operation: "linear_interpolation_backward",
                reason: error.to_string(),
            })
        })?;
        let backend = B::default();
        if let Some(Some(gradient)) = input_grads.first() {
            coeus_ops::add_assign(gradient.write(), &updates.image, &backend)?;
        }
        if let Some(Some(gradient)) = input_grads.get(1) {
            coeus_ops::add_assign(gradient.write(), &updates.grid, &backend)?;
        }
        Ok(())
    }
}

/// Sample a 2-D or 3-D image while tracking image and coordinate gradients.
///
/// `D` selects the spatial dimension and `P` is a zero-sized boundary policy.
///
/// # Errors
///
/// Returns [`InterpolationFailure`] when image or grid shape violates the
/// dimension-generic interpolation contract.
pub fn linear_interpolation<T, const D: usize, B, P>(
    image: &Var<T, B>,
    grid: &Var<T, B>,
    policy: P,
) -> Result<Var<T, B>, InterpolationFailure<B::Error>>
where
    T: Float + FloatElement,
    B: Backend + coeus_ops::BackendOps<T> + Default,
    P: BoundaryPolicy + Send + Sync + 'static,
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let output =
        coeus_ops::linear_interpolation::<T, D, _, P>(&image.tensor, &grid.tensor, policy)?;
    let requires_grad =
        crate::grad_mode::should_track_var(image) || crate::grad_mode::should_track_var(grid);
    if !requires_grad {
        return Var::new(output, false).map_err(InterpolationFailure::Backend);
    }

    let backend = B::default();
    Var::from_tracked_op(output, requires_grad, &backend, |output_grad| {
        LinearInterpolationNode::<T, D, B, P> {
            output_grad,
            inputs: vec![image.clone(), grid.clone()],
            image: image.tensor.clone(),
            grid: grid.tensor.clone(),
            policy: PhantomData,
        }
    })
    .map_err(InterpolationFailure::Backend)
}
