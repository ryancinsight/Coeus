use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct PermuteNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub inv_dims: Vec<usize>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for PermuteNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "permute"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let permuted_grad = grad_out.permute(&self.inv_dims);
            let gl = g.write();
            coeus_ops::add_assign(gl, &permuted_grad, &backend);
        }
    }
}

/// Tracked permute operation.
#[inline]
pub fn permute<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    dims: &[usize],
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = x.tensor.permute(dims);

    let requires_grad = crate::grad_mode::should_track_var(x);
    if !requires_grad {
        return Var::new(out_tensor, false);
    }

    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(
        out_tensor.shape_cloned(),
        &backend,
    )));
    let grad = Some(output_grad.clone());

    // Compute inverse permutation
    let mut inv_dims = vec![0; dims.len()];
    for (i, &d) in dims.iter().enumerate() {
        inv_dims[d] = i;
    }

    let node = PermuteNode {
        output_grad,
        inputs: vec![x.clone()],
        inv_dims,
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

/// Tracked general transpose operation. Swaps `dim0` and `dim1`.
#[inline]
pub fn transpose<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    dim0: usize,
    dim1: usize,
) -> Var<T, B> {
    let ndim = x.tensor.ndim();
    assert!(
        dim0 < ndim && dim1 < ndim,
        "transpose: dimensions out of bounds"
    );
    if dim0 == dim1 {
        return x.clone();
    }
    let mut dims: Vec<usize> = (0..ndim).collect();
    dims.swap(dim0, dim1);
    permute(x, &dims)
}
