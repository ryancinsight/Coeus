//! Differentiable dimension-generic coordinate-grid interpolation.

use crate::{grad_buffer::GradBuffer, node::BackwardNode, var::Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::{
    linear_interpolation_backward, BoundaryPolicy, Dimension, InterpolationError, Replicate,
    SupportedDimension,
};
use coeus_tensor::Tensor;
use std::{marker::PhantomData, sync::Arc};

/// Reverse-mode node for linear sampling.
pub struct LinearInterpolationNode<const D: usize, B, P = Replicate>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    P: BoundaryPolicy,
{
    /// Accumulated output gradient.
    pub output_grad: Arc<GradBuffer<f32, B>>,
    /// Image and sampling-grid variables.
    pub inputs: Vec<Var<f32, B>>,
    /// Saved image values required by the coordinate derivative.
    pub image: Tensor<f32, B>,
    /// Saved sampling coordinates required by both derivatives.
    pub grid: Tensor<f32, B>,
    policy: PhantomData<P>,
}

impl<const D: usize, B, P> BackwardNode<f32, B> for LinearInterpolationNode<D, B, P>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    P: BoundaryPolicy + Send + Sync + 'static,
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn op_name(&self) -> &'static str {
        "linear_interpolation"
    }

    fn output_grad(&self) -> &Arc<GradBuffer<f32, B>> {
        &self.output_grad
    }

    fn inputs(&self) -> &[Var<f32, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<f32, B>, input_grads: &[Option<Arc<GradBuffer<f32, B>>>]) {
        let updates = linear_interpolation_backward::<D, _, P>(
            &self.image,
            &self.grid,
            grad_out,
            P::default(),
        )
        .expect("invariant: forward validation fixes backward shapes");
        let backend = B::default();
        if let Some(Some(gradient)) = input_grads.first() {
            coeus_ops::add_assign(gradient.write(), &updates.image, &backend);
        }
        if let Some(Some(gradient)) = input_grads.get(1) {
            coeus_ops::add_assign(gradient.write(), &updates.grid, &backend);
        }
    }
}

/// Sample a 2-D or 3-D image while tracking image and coordinate gradients.
///
/// `D` selects the spatial dimension and `P` is a zero-sized boundary policy.
///
/// # Errors
///
/// Returns [`InterpolationError`] when image or grid shape violates the
/// dimension-generic interpolation contract.
pub fn linear_interpolation<const D: usize, B, P>(
    image: &Var<f32, B>,
    grid: &Var<f32, B>,
    policy: P,
) -> Result<Var<f32, B>, InterpolationError>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    P: BoundaryPolicy + Send + Sync + 'static,
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let output = coeus_ops::linear_interpolation::<D, _, P>(&image.tensor, &grid.tensor, policy)?;
    let requires_grad =
        crate::grad_mode::should_track_var(image) || crate::grad_mode::should_track_var(grid);
    if !requires_grad {
        return Ok(Var::new(output, false));
    }

    let backend = B::default();
    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(output.shape(), &backend)));
    let node = LinearInterpolationNode::<D, B, P> {
        output_grad: output_grad.clone(),
        inputs: vec![image.clone(), grid.clone()],
        image: image.tensor.clone(),
        grid: grid.tensor.clone(),
        policy: PhantomData,
    };
    Ok(Var {
        tensor: output,
        grad: Some(output_grad),
        creator: Some(Arc::new(node)),
    })
}
