use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for multi-label margin loss.
///
/// PyTorch `MultiLabelMarginLoss` with `reduction="mean"`:
/// x: `(N, C)`, target: `(N, C)` with `-1` = ignore padding.
/// Loss per sample: sum over valid targets `t` of sum over `j != t` of
/// `max(0, 1 - (x[t] - x[j]))`, normalized by `(N * C)`.
pub struct MultiLabelMarginLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Gradient buffer for the output scalar.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Input x copied to host: `(N, C)` contiguous.
    pub x_host: Vec<T>,
    /// Target indices: `(N, C)` with `-1` = ignore, as `isize`.
    pub target: Vec<isize>,
    /// Number of samples.
    pub n: usize,
    /// Number of classes.
    pub c: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for MultiLabelMarginLossNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "multi_label_margin_loss"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
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
        let scale = g_out / T::from_f64((self.n * self.c) as f64);

        if let Some(Some(ref g)) = input_grads.get(0) {
            let mut dx = vec![T::zero(); self.n * self.c];
            for i in 0..self.n {
                let base = i * self.c;
                for k in 0..self.c {
                    let t_raw = self.target[base + k];
                    if t_raw < 0 {
                        continue;
                    }
                    let t = t_raw as usize;
                    for j in 0..self.c {
                        if j == t {
                            continue;
                        }
                        let diff = self.x_host[base + t] - self.x_host[base + j];
                        if diff < T::one() {
                            // d/d x[t] (1 - (x[t] - x[j])) = -1
                            dx[base + t] = dx[base + t] - scale;
                            // d/d x[j] (1 - (x[t] - x[j])) = +1
                            dx[base + j] = dx[base + j] + scale;
                        }
                    }
                }
            }
            let grad_tensor = Tensor::from_slice_on([self.n, self.c], &dx, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked multi-label margin loss (PyTorch `MultiLabelMarginLoss` with
/// `reduction="mean"`).
///
/// `x`: shape `(N, C)`, target: `(N, C)` where `target[i][j] >= 0` are valid
/// class indices and `-1` means ignore. Returns a scalar `Var`.
pub fn multi_label_margin_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    target: &[isize],
) -> Var<T, B> {
    let shape = x.tensor.shape();
    assert_eq!(shape.len(), 2, "x must be 2D [batch_size, num_classes]");
    let n = shape[0];
    let c = shape[1];
    assert_eq!(target.len(), n * c, "target length must match N*C");

    let backend = B::default();

    let x_cont;
    let x_raw = if x.tensor.is_contiguous() && x.tensor.layout().offset() == 0 {
        &x.tensor
    } else {
        x_cont = x.tensor.to_contiguous_on(&backend);
        &x_cont
    };

    let x_host: Vec<T> = if let Some(s) = x_raw.storage().try_as_slice() {
        s[..n * c].to_vec()
    } else {
        let mut v = vec![T::zero(); n * c];
        backend.copy_to_host(x_raw.storage(), &mut v);
        v
    };

    let zero = T::zero();
    let one = T::one();
    let mut loss_val = zero;

    for i in 0..n {
        let base = i * c;
        for k in 0..c {
            let t_raw = target[base + k];
            if t_raw < 0 {
                continue;
            }
            let t = t_raw as usize;
            for j in 0..c {
                if j == t {
                    continue;
                }
                let diff = x_host[base + t] - x_host[base + j];
                let hinge = if one - diff > zero { one - diff } else { zero };
                loss_val = loss_val + hinge;
            }
        }
    }
    loss_val = loss_val / T::from_f64((n * c) as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = crate::grad_mode::should_track_var(x);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            x.tensor.shape(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = MultiLabelMarginLossNode {
            output_grad,
            inputs: vec![x.clone()],
            x_host,
            target: target.to_vec(),
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
