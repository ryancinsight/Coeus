use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for cosine embedding loss.
///
/// Forward and backward stay on the selected provider: the node retains the
/// provider-resident per-row dot/norm denominators and the target-weight
/// factor rather than host `Vec<T>` payloads. The `y: &[T]` host slice is a
/// boundary upload.
pub struct CosineEmbeddingLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident target labels (1 or -1), shape `[N]`.
    pub y: Tensor<T, B>,
    /// Provider-resident `1 / den` per row, shape `[N, 1]`.
    pub inv_den: Tensor<T, B>,
    /// Provider-resident `dot / norm1_sq` per row, shape `[N, 1]`.
    pub dot_over_n1sq: Tensor<T, B>,
    /// Provider-resident `dot / norm2_sq` per row, shape `[N, 1]`.
    pub dot_over_n2sq: Tensor<T, B>,
    /// Provider-resident row weight `w_i` (nonzero where the hinge is active),
    /// shape `[N, 1]`.
    pub row_weight: Tensor<T, B>,
    /// Margin for dissimilar pairs.
    pub margin: T,
    /// Batch size.
    pub n: usize,
    /// Embedding dimension.
    pub d: usize,
    /// Provider-resident mean scale `1 / batch_size`.
    pub mean_scale: Tensor<T, B>,
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

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        let need_g1 = input_grads.first().and_then(|g| g.as_ref()).is_some();
        let need_g2 = input_grads.get(1).and_then(|g| g.as_ref()).is_some();
        if !need_g1 && !need_g2 {
            return Ok(());
        }
        // dg1 = w * scale * (x2 - (dot/n1_sq) * x1) / den; dg2 similarly.
        // All factors are per-row broadcast; composed from provider ops only.
        let scale = coeus_ops::mul(grad_out, &self.mean_scale, &backend);
        let row_scale = self.row_weight.reshape([self.n, 1]).broadcast([self.n, 1]);
        let combined = coeus_ops::mul(&row_scale, &scale.broadcast([self.n, 1]), &backend);
        let combined = coeus_ops::mul(&combined, &self.inv_den, &backend);

        let x1 = &self.inputs[0].tensor;
        let x2 = &self.inputs[1].tensor;
        if need_g1 {
            let proj = coeus_ops::mul(&self.dot_over_n1sq, x1, &backend);
            let diff = coeus_ops::sub(x2, &proj, &backend);
            let dg1 = coeus_ops::mul(&diff, &combined, &backend);
            if let Some(Some(ref g)) = input_grads.first() {
                coeus_ops::add_assign(g.write(), &dg1, &backend)?;
            }
        }
        if need_g2 {
            let proj = coeus_ops::mul(&self.dot_over_n2sq, x2, &backend);
            let diff = coeus_ops::sub(x1, &proj, &backend);
            let dg2 = coeus_ops::mul(&diff, &combined, &backend);
            if let Some(Some(ref g)) = input_grads.get(1) {
                coeus_ops::add_assign(g.write(), &dg2, &backend)?;
            }
        }
        Ok(())
    }
}

/// Tracked Cosine Embedding Loss.
/// `x1`: `[N, D]`, `x2`: `[N, D]`, `y`: `[N]` (elements 1 or -1).
/// The complete forward and backward computation stays on the selected
/// provider; no input-sized host staging occurs beyond the `y` boundary upload.
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

    // Per-row dot and squared norms, all on-provider.
    let dot = coeus_ops::sum_axis(
        &coeus_ops::mul(&x1.tensor, &x2.tensor, &backend),
        1,
        &backend,
    )
    .expect("invariant: validated [N, D] cosine dot reduction");
    let norm1_sq = coeus_ops::sum_axis(
        &coeus_ops::mul(&x1.tensor, &x1.tensor, &backend),
        1,
        &backend,
    )
    .expect("invariant: validated [N, D] norm1 reduction");
    let norm2_sq = coeus_ops::sum_axis(
        &coeus_ops::mul(&x2.tensor, &x2.tensor, &backend),
        1,
        &backend,
    )
    .expect("invariant: validated [N, D] norm2 reduction");

    let eps = T::from_f64(1e-8);
    let eps_tensor = Tensor::full_on([n, 1], eps, &backend);
    let dot_col = dot.reshape([n, 1]);
    let n1_col = norm1_sq.reshape([n, 1]);
    let n2_col = norm2_sq.reshape([n, 1]);
    let den_sq = coeus_ops::mul(&n1_col, &n2_col, &backend);
    let den_sq_safe = coeus_ops::where_cond(
        &coeus_ops::gt(&den_sq, &eps_tensor, &backend),
        &den_sq,
        &eps_tensor,
        &backend,
    )
    .expect("cosine_embedding_loss: denom clamp");
    let den = coeus_ops::sqrt(&den_sq_safe, &backend);
    let inv_den = coeus_ops::div(&Tensor::full_on([n, 1], T::one(), &backend), &den, &backend);
    let cos = coeus_ops::mul(&dot_col, &inv_den, &backend);
    let dot_over_n1sq = coeus_ops::div(&dot_col, &n1_col, &backend);
    let dot_over_n2sq = coeus_ops::div(&dot_col, &n2_col, &backend);

    // Loss per row: y == 1 → 1 - cos; else relu(cos - margin).
    let y_tensor = Tensor::from_slice_on([n], y, &backend).reshape([n, 1]);
    let ones = Tensor::full_on([n, 1], T::one(), &backend);
    let y_is_one = coeus_ops::eq(&y_tensor, &ones, &backend);
    let pos_loss = coeus_ops::sub(&ones, &cos, &backend);
    let neg_diff = coeus_ops::sub(&cos, &Tensor::full_on([n, 1], margin, &backend), &backend);
    let neg_loss = coeus_ops::relu(&neg_diff, &backend);
    let per_row = coeus_ops::where_cond(&y_is_one, &pos_loss, &neg_loss, &backend)
        .expect("cosine_embedding_loss: branch select");
    let loss = coeus_ops::mean_axis(&per_row.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty cosine-embedding reduction has axis zero");

    // Row weight for backward: y == 1 → -1; else +1 only where cos > margin.
    let neg_ones = coeus_ops::neg(&ones, &backend);
    let active = coeus_ops::gt(&cos, &Tensor::full_on([n, 1], margin, &backend), &backend);
    let active_ones = coeus_ops::where_cond(
        &active,
        &ones,
        &Tensor::zeros_on([n, 1], &backend),
        &backend,
    )
    .expect("cosine_embedding_loss: hinge mask");
    let row_weight = coeus_ops::where_cond(&y_is_one, &neg_ones, &active_ones, &backend)
        .expect("cosine_embedding_loss: weight select");

    let requires_grad =
        crate::grad_mode::should_track_var(x1) || crate::grad_mode::should_track_var(x2);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };

    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = CosineEmbeddingLossNode {
            output_grad,
            inputs: vec![x1.clone(), x2.clone()],
            y: y_tensor.reshape([n]),
            inv_den,
            dot_over_n1sq,
            dot_over_n2sq,
            row_weight,
            margin,
            n,
            d,
            mean_scale: Tensor::full_on([1], T::one() / T::from_f64(n as f64), &backend),
        };
        Arc::new(node) as Arc<dyn BackwardNode<T, B>>
    });

    Var {
        tensor: loss,
        grad,
        creator,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;

    #[test]
    fn cosine_embedding_forward_matches_reference() {
        // x1 = [[1, 0], [1, 0]], x2 = [[1, 0], [0, 1]], y = [1, -1], margin = 0.5:
        //   row0 cos = 1 → loss 1 - 1 = 0
        //   row1 cos = 0 → relu(0 - 0.5) = 0
        //   loss = 0.
        let x1 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0, 0.0, 1.0, 0.0]),
            true,
        );
        let x2 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0, 0.0, 0.0, 1.0]),
            true,
        );
        let y = [1.0, -1.0];
        let loss = cosine_embedding_loss(&x1, &x2, &y, 0.5);
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_embedding_forward_dissimilar_active() {
        // x1 = [[1, 0]], x2 = [[-1, 0]], y = [-1], margin = 0.5:
        //   cos = -1 → relu(-1 - 0.5) = 0 → loss 0.
        // x2 = [[0, 1]]: cos = 0 → relu(0 - 0.5) = 0.
        let x1 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0, 0.0]),
            true,
        );
        let x2 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[-1.0, 0.0]),
            true,
        );
        let y = [-1.0];
        let loss = cosine_embedding_loss(&x1, &x2, &y, 0.5);
        assert!((loss.tensor.as_slice()[0] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_embedding_backward_matches_analytic() {
        // x1 = [[1, 0]], x2 = [[1, 0]], y = [1]: cos = 1, loss 0.
        // d/dx1 = -w * scale * (x2 - dot/n1_sq * x1)/den with w = -1, scale=1/1:
        //   dot = 1, n1_sq = 1, den = 1 → dg1 = -(1 - 1·1)/1 = 0.
        let x1 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0, 0.0]),
            true,
        );
        let x2 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0, 0.0]),
            true,
        );
        let y = [1.0];
        let loss = cosine_embedding_loss(&x1, &x2, &y, 0.5);
        loss.backward().expect("invariant: backward completes");
        let g1 = x1.grad().expect("x1 must receive a gradient");
        let g2 = x2.grad().expect("x2 must receive a gradient");
        for (i, &g) in g1.as_slice().iter().enumerate() {
            assert!((g - 0.0).abs() < 1e-12, "cosine dg1[{i}] = {g}, expected 0");
        }
        for (i, &g) in g2.as_slice().iter().enumerate() {
            assert!((g - 0.0).abs() < 1e-12, "cosine dg2[{i}] = {g}, expected 0");
        }
    }

    #[test]
    fn cosine_embedding_backward_nonzero_case() {
        // x1 = [[1, 0]], x2 = [[0, 1]], y = [1]: cos = 0, loss 1 - 0 = 1.
        // d/dx1 = -1·(x2 - dot/n1_sq·x1)/den = -(0,1)/1 = [0, -1].
        let x1 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0, 0.0]),
            true,
        );
        let x2 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 1.0]),
            true,
        );
        let y = [1.0];
        let loss = cosine_embedding_loss(&x1, &x2, &y, 0.5);
        loss.backward().expect("invariant: backward completes");
        let g1 = x1.grad().expect("x1 must receive a gradient");
        let expected = [0.0, -1.0];
        for (i, (&g, &e)) in g1.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-9, "cosine dg1[{i}] = {g}, expected {e}");
        }
    }

    #[test]
    #[should_panic(expected = "same shape")]
    fn cosine_embedding_rejects_shape_mismatch() {
        let x1 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0, 0.0]),
            true,
        );
        let x2 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[1.0, 0.0, 0.0]),
            true,
        );
        let y = [1.0];
        let _ = cosine_embedding_loss(&x1, &x2, &y, 0.5);
    }
}
