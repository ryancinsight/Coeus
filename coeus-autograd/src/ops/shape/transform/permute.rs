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

/// Swap two axes (`torch.swapaxes` / `np.swapaxes`) — a named alias for
/// [`transpose`].
#[inline]
#[must_use]
pub fn swapaxes<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    axis0: usize,
    axis1: usize,
) -> Var<T, B> {
    transpose(x, axis0, axis1)
}

/// Move dimension `source` to position `dest` (`torch.movedim` / `np.moveaxis`),
/// preserving the relative order of the remaining dimensions. Differentiable via
/// [`permute`].
///
/// # Panics
/// If `source` or `dest` is out of range.
#[must_use]
pub fn movedim<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    source: usize,
    dest: usize,
) -> Var<T, B> {
    let ndim = x.tensor.ndim();
    assert!(
        source < ndim && dest < ndim,
        "movedim: source {source} / dest {dest} out of range for rank {ndim}"
    );
    if source == dest {
        return x.clone();
    }
    let mut order: Vec<usize> = (0..ndim).filter(|&d| d != source).collect();
    order.insert(dest, source);
    permute(x, &order)
}

#[cfg(test)]
mod movedim_tests {
    use super::*;
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn movedim_reorders_axes_and_backprops() {
        // [2,3,4]; movedim(0,2) -> [3,4,2] with out[i,j,k] == in[k,i,j].
        let data: Vec<f64> = (0..24).map(|v| v as f64).collect();
        let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([2, 3, 4], &data), true);
        let moved = movedim(&x, 0, 2);
        assert_eq!(moved.tensor.shape(), &[3, 4, 2]);
        let out = moved.tensor.to_contiguous();
        let o = out.as_slice();
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..2 {
                    assert_eq!(
                        o[(i * 4 + j) * 2 + k],
                        data[k * 12 + i * 4 + j],
                        "[{i},{j},{k}]"
                    );
                }
            }
        }
        moved.backward();
        assert_eq!(x.grad().unwrap().shape(), &[2, 3, 4]);
    }

    #[test]
    fn swapaxes_matches_transpose() {
        let data: Vec<f64> = (0..6).map(|v| v as f64).collect();
        let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([2, 3], &data), false);
        let s = swapaxes(&x, 0, 1);
        let t = transpose(&x, 0, 1);
        assert_eq!(s.tensor.shape(), t.tensor.shape());
        assert_eq!(
            s.tensor.to_contiguous().as_slice(),
            t.tensor.to_contiguous().as_slice()
        );
    }
}
