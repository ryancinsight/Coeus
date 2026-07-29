use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for the multi-class margin loss.
pub struct MultiMarginNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Per-element `d(loss)/d(x_ij)` assuming a unit upstream gradient, `[N*C]`.
    pub grad_unit: Vec<T>,
    /// Batch size.
    pub n: usize,
    /// Number of classes.
    pub c: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for MultiMarginNode<T, B> {
    fn op_name(&self) -> &'static str {
        "multi_margin"
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
            let mut host_grad = [T::zero()];
            let temp_grad;
            let grad_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
                grad_out
            } else {
                temp_grad = grad_out.to_contiguous_on(&backend);
                &temp_grad
            };
            backend.copy_to_host(grad_cont.storage(), &mut host_grad);
            let g_out = host_grad[0];

            let mut d_x = vec![T::zero(); self.n * self.c];
            for (i, gv) in d_x.iter_mut().enumerate() {
                *gv = g_out * self.grad_unit[i];
            }
            let grad_tensor = Tensor::from_slice_on([self.n, self.c], &d_x, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)
                .expect("autograd gradient accumulation");
        }
    }
}

/// Tracked multi-class margin loss (PyTorch `MultiMarginLoss`, `reduction="mean"`):
/// `mean_i (1/C) sum_{j != y_i} max(0, margin - x[i,y_i] + x[i,j])^p`.
///
/// `x`: `[N, C]` scores, `targets`: `[N]` class indices, `p >= 1`, `margin`.
/// Gradient (unit upstream): for `j != y_i` with `m_ij = margin - x[i,y_i] + x[i,j] > 0`,
/// `d/d x_ij = p*m_ij^(p-1) / (N*C)`; the target column accumulates the negation of
/// every active sibling term.
pub fn multi_margin<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    targets: &[usize],
    p: T,
    margin: T,
) -> Var<T, B> {
    let backend = B::default();
    let shape = x.tensor.shape();
    assert_eq!(shape.len(), 2, "multi_margin expects 2D [N, C] input");
    let n = shape[0];
    let c = shape[1];
    assert_eq!(targets.len(), n, "targets length must match batch size");

    let cont;
    let x_raw = if x.tensor.is_contiguous() && x.tensor.layout().offset() == 0 {
        &x.tensor
    } else {
        cont = x.tensor.to_contiguous_on(&backend);
        &cont
    };
    let host: std::borrow::Cow<[T]> = if let Some(s) = x_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n * c])
    } else {
        let mut v = vec![T::zero(); n * c];
        backend.copy_to_host(x_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    let one = T::one();
    let p_minus_one = p - one;
    let inv_nc = one / T::from_f64((n * c) as f64);
    let mut grad_unit = vec![T::zero(); n * c];
    let mut total = T::zero();
    for i in 0..n {
        let y = targets[i];
        assert!(y < c, "target index out of bounds");
        let base = i * c;
        let xy = host[base + y];
        let mut diag_coef = T::zero();
        for j in 0..c {
            if j == y {
                continue;
            }
            let m = margin - xy + host[base + j];
            if m > T::zero() {
                // loss contribution m^p; derivative coef p*m^(p-1).
                total += m.powf(p);
                let coef = p * m.powf(p_minus_one);
                grad_unit[base + j] = coef * inv_nc;
                diag_coef += coef;
            }
        }
        grad_unit[base + y] = (T::zero() - diag_coef) * inv_nc;
    }
    let loss_val = total * inv_nc;

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = crate::grad_mode::should_track_var(x);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = MultiMarginNode {
            output_grad,
            inputs: vec![x.clone()],
            grad_unit,
            n,
            c,
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
