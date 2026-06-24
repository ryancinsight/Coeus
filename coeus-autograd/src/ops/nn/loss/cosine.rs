use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct CosineEmbeddingLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub x1_host: Vec<T>,
    pub x2_host: Vec<T>,
    pub y: Vec<T>,
    pub margin: T,
    pub n: usize,
    pub d: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for CosineEmbeddingLossNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "cosine_embedding_loss"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        let need_g1 = input_grads.get(0).and_then(|g| g.as_ref()).is_some();
        let need_g2 = input_grads.get(1).and_then(|g| g.as_ref()).is_some();
        if !need_g1 && !need_g2 {
            return;
        }

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
        let n_t = T::from_f64(self.n as f64);
        let scale = g_out / n_t;
        let eps = T::from_f64(1e-8);

        let mut dg1 = vec![T::zero(); self.n * self.d];
        let mut dg2 = vec![T::zero(); self.n * self.d];

        for i in 0..self.n {
            let offset = i * self.d;
            let mut dot = T::zero();
            let mut norm1_sq = T::zero();
            let mut norm2_sq = T::zero();
            for j in 0..self.d {
                let val1 = self.x1_host[offset + j];
                let val2 = self.x2_host[offset + j];
                dot = dot + val1 * val2;
                norm1_sq = norm1_sq + val1 * val1;
                norm2_sq = norm2_sq + val2 * val2;
            }
            let norm1 = norm1_sq.sqrt();
            let norm2 = norm2_sq.sqrt();
            let den = if norm1 * norm2 > eps {
                norm1 * norm2
            } else {
                eps
            };
            let cos = dot / den;

            let y_val = self.y[i];
            let target_is_one = y_val == T::one();
            let w_i = if target_is_one {
                T::zero() - T::one()
            } else {
                if cos > self.margin {
                    T::one()
                } else {
                    T::zero()
                }
            };

            if w_i != T::zero() {
                let n1_sq_safe = if norm1_sq > eps { norm1_sq } else { eps };
                let n2_sq_safe = if norm2_sq > eps { norm2_sq } else { eps };

                for j in 0..self.d {
                    let val1 = self.x1_host[offset + j];
                    let val2 = self.x2_host[offset + j];
                    let g1_val = w_i * scale * (val2 - (dot / n1_sq_safe) * val1) / den;
                    dg1[offset + j] = g1_val;

                    let g2_val = w_i * scale * (val1 - (dot / n2_sq_safe) * val2) / den;
                    dg2[offset + j] = g2_val;
                }
            }
        }

        if let Some(Some(ref g)) = input_grads.get(0) {
            let grad_tensor = Tensor::from_slice_on([self.n, self.d], &dg1, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let grad_tensor = Tensor::from_slice_on([self.n, self.d], &dg2, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Cosine Embedding Loss.
/// x1: `[N, D]`, x2: `[N, D]`, y: `[N]` (elements 1 or -1).
pub fn cosine_embedding_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    y: &[T],
    margin: T,
) -> Var<T, B> {
    let backend = B::default();
    let n = x1.tensor.shape()[0];
    let d = x1.tensor.shape()[1];
    assert_eq!(
        x2.tensor.shape(),
        x1.tensor.shape(),
        "cosine_embedding_loss: x1 and x2 must have same shape"
    );
    assert_eq!(
        y.len(),
        n,
        "cosine_embedding_loss: y must have length equal to batch size"
    );

    let x1_cont;
    let x1_raw = if x1.tensor.is_contiguous() && x1.tensor.layout().offset() == 0 {
        &x1.tensor
    } else {
        x1_cont = x1.tensor.to_contiguous_on(&backend);
        &x1_cont
    };
    let x2_cont;
    let x2_raw = if x2.tensor.is_contiguous() && x2.tensor.layout().offset() == 0 {
        &x2.tensor
    } else {
        x2_cont = x2.tensor.to_contiguous_on(&backend);
        &x2_cont
    };

    let numel = n * d;
    let x1_host: Vec<T> = if let Some(s) = x1_raw.storage().try_as_slice() {
        s[..numel].to_vec()
    } else {
        let mut v = vec![T::zero(); numel];
        backend.copy_to_host(x1_raw.storage(), &mut v);
        v
    };
    let x2_host: Vec<T> = if let Some(s) = x2_raw.storage().try_as_slice() {
        s[..numel].to_vec()
    } else {
        let mut v = vec![T::zero(); numel];
        backend.copy_to_host(x2_raw.storage(), &mut v);
        v
    };

    let eps = T::from_f64(1e-8);
    let mut loss_val = T::zero();
    for i in 0..n {
        let offset = i * d;
        let mut dot = T::zero();
        let mut norm1_sq = T::zero();
        let mut norm2_sq = T::zero();
        for j in 0..d {
            let val1 = x1_host[offset + j];
            let val2 = x2_host[offset + j];
            dot = dot + val1 * val2;
            norm1_sq = norm1_sq + val1 * val1;
            norm2_sq = norm2_sq + val2 * val2;
        }
        let norm1 = norm1_sq.sqrt();
        let norm2 = norm2_sq.sqrt();
        let den = if norm1 * norm2 > eps {
            norm1 * norm2
        } else {
            eps
        };
        let cos = dot / den;
        let y_val = y[i];
        let target_is_one = y_val == T::one();
        let item_loss = if target_is_one {
            T::one() - cos
        } else {
            let diff = cos - margin;
            if diff > T::zero() {
                diff
            } else {
                T::zero()
            }
        };
        loss_val = loss_val + item_loss;
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = x1.grad.is_some() || x2.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = CosineEmbeddingLossNode {
            output_grad,
            inputs: vec![x1.clone(), x2.clone()],
            x1_host,
            x2_host,
            y: y.to_vec(),
            margin,
            n,
            d,
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
