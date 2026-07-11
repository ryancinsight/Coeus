//! Differentiable coordinate-grid interpolation.

use crate::{grad_buffer::GradBuffer, node::BackwardNode, var::Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::{trilinear_interpolation_backward, InterpolationError};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Reverse-mode node for trilinear sampling.
pub struct TrilinearInterpolationNode<B>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
{
    /// Accumulated output gradient.
    pub output_grad: Arc<GradBuffer<f32, B>>,
    /// Image and sampling-grid variables.
    pub inputs: Vec<Var<f32, B>>,
    /// Saved image values required by the coordinate derivative.
    pub image: Tensor<f32, B>,
    /// Saved sampling coordinates required by both derivatives.
    pub grid: Tensor<f32, B>,
}

impl<B> BackwardNode<f32, B> for TrilinearInterpolationNode<B>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn op_name(&self) -> &'static str {
        "trilinear_interpolation"
    }

    fn output_grad(&self) -> &Arc<GradBuffer<f32, B>> {
        &self.output_grad
    }

    fn inputs(&self) -> &[Var<f32, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<f32, B>, input_grads: &[Option<Arc<GradBuffer<f32, B>>>]) {
        let updates = trilinear_interpolation_backward(&self.image, &self.grid, grad_out)
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

/// Sample an image at a voxel-coordinate grid while tracking both gradients.
///
/// # Errors
///
/// Returns [`InterpolationError`] when the image or grid violates the rank-5
/// trilinear contract.
pub fn trilinear_interpolation<B>(
    image: &Var<f32, B>,
    grid: &Var<f32, B>,
) -> Result<Var<f32, B>, InterpolationError>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let output = coeus_ops::trilinear_interpolation(&image.tensor, &grid.tensor)?;
    let requires_grad =
        crate::grad_mode::should_track_var(image) || crate::grad_mode::should_track_var(grid);
    if !requires_grad {
        return Ok(Var::new(output, false));
    }

    let backend = B::default();
    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(output.shape(), &backend)));
    let node = TrilinearInterpolationNode {
        output_grad: output_grad.clone(),
        inputs: vec![image.clone(), grid.clone()],
        image: image.tensor.clone(),
        grid: grid.tensor.clone(),
    };
    Ok(Var {
        tensor: output,
        grad: Some(output_grad),
        creator: Some(Arc::new(node)),
    })
}
