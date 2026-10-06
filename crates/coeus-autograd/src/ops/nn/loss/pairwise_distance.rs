use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for the row-wise p-norm pairwise distance.
///
/// Forward and backward stay on the selected provider: the node retains the
/// provider-resident `s = sum_k |x1 - x2 + eps|^p` row sums and the `[N,D]`
/// signed powered magnitudes rather than host `Vec<T>` payloads.
pub struct PairwiseDistanceNode<
    T: Float,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
> {
    /// Accumulated gradient buffer for the output of this node (shape `[N]`).
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident `s^(1/p - 1)` row scales (shape `[N, 1]`).
    pub row_scale: Tensor<T, B>,
    /// Provider-resident `sign(diff) * |diff|^(p-1)` (shape `[N, D]`).
    pub grad_unit: Tensor<T, B>,
    /// Number of rows `N`.
    pub rows: usize,
    /// Feature dimension `D`.
    pub feat: usize,
    /// Input shape `[N, D]` for gradient reconstruction.
    pub shape: coeus_core::Shape,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default>
    BackwardNode<T, B> for PairwiseDistanceNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "pairwise_distance"
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
        let backend = B::default();
        let want_x1 = matches!(input_grads.first(), Some(Some(_)));
        let want_x2 = matches!(input_grads.get(1), Some(Some(_)));
        if !want_x1 && !want_x2 {
            return Ok(());
        }
        // grad_rows broadcasts [N] over [N, D]; d/dx1 = grad_unit * row_scale
        // * grad_rows. All on-provider.
        let row_grad = grad_out.reshape([self.rows, 1]);
        let broadcast = coeus_ops::mul(&self.grad_unit, &self.row_scale, &backend);
        let dx1 = coeus_ops::mul(
            &broadcast,
            &row_grad.broadcast(self.shape.clone()),
            &backend,
        );

        if want_x1 {
            if let Some(Some(ref g)) = input_grads.first() {
                coeus_ops::add_assign(g.write(), &dx1, &backend)?;
            }
        }
        if want_x2 {
            if let Some(Some(ref g)) = input_grads.get(1) {
                let dx2 = coeus_ops::neg(&dx1, &backend);
                coeus_ops::add_assign(g.write(), &dx2, &backend)?;
            }
        }
        Ok(())
    }
}

/// Tracked row-wise p-norm pairwise distance (PyTorch `PairwiseDistance`):
/// for inputs `x1, x2` of shape `[N, D]`, returns a `[N]` vector with
/// `out_i = (sum_k |x1_ik - x2_ik + eps|^p)^(1/p)`. Matches PyTorch's
/// `torch.nn.functional.pairwise_distance` (`at::norm(x1 - x2 + eps, p)`)
/// exactly: `eps` is added to the difference, keeping the summed norm
/// strictly positive so the `s^(1/p - 1)` gradient factor stays finite.
///
/// The complete forward and backward computation stays on the selected
/// provider; no input-sized host staging occurs.
pub fn pairwise_distance<
    T: Float,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    p: T,
    eps: T,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        x1.tensor.shape(),
        x2.tensor.shape(),
        "pairwise_distance requires x1 and x2 to have identical shapes"
    );
    let shape_ref = x1.tensor.shape();
    assert_eq!(
        shape_ref.len(),
        2,
        "pairwise_distance expects 2D [N, D] inputs"
    );
    let rows = shape_ref[0];
    let feat = shape_ref[1];
    let shape = x1.tensor.shape_cloned();

    // diff = x1 - x2 + eps; out_i = norm_p_axis(diff, p, axis=1) — the same
    // row-wise p-norm as PyTorch's at::norm(x1 - x2 + eps, p), composed from
    // provider abs/pow_scalar/sum_axis with the exact dtype bounds.
    let diff = coeus_ops::sub(&x1.tensor, &x2.tensor, &backend);
    let shifted = coeus_ops::add(
        &diff,
        &Tensor::full_on(shape.clone(), eps, &backend),
        &backend,
    );
    let row_norm = coeus_ops::norm_p_axis(&shifted, p, 1, &backend);
    let out = row_norm.reshape([rows]);

    // Backward factors: row_scale = out^(1-p) reshaped to [N, 1] (the
    // `s^(1/p-1)` factor equals `out^(1-p)` since out = s^(1/p));
    // grad_unit = sign(diff) * |diff|^(p-1).
    let one_minus_p = T::one() - p;
    let row_scale = coeus_ops::pow_scalar(&row_norm, one_minus_p, &backend);
    let row_scale = row_scale.reshape([rows, 1]);
    let p_minus_one = p - T::one();
    let magnitudes = coeus_ops::abs(&shifted, &backend);
    let grad_unit = coeus_ops::mul(
        &coeus_ops::sign(&shifted, &backend),
        &coeus_ops::pow_scalar(&magnitudes, p_minus_one, &backend),
        &backend,
    );

    let out_tensor = out;
    let requires_grad =
        crate::grad_mode::should_track_var(x1) || crate::grad_mode::should_track_var(x2);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            [rows],
            &backend,
        ))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = PairwiseDistanceNode {
            output_grad,
            inputs: vec![x1.clone(), x2.clone()],
            row_scale,
            grad_unit,
            rows,
            feat,
            shape,
        };
        Arc::new(node) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;

    #[test]
    fn pairwise_distance_forward_matches_reference() {
        // x1 = [[3, 4]], x2 = [[0, 0]], p = 2, eps = 1e-6:
        //   diff = [3, 4], s = 9 + 16 = 25, out = 5.
        let x1 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[3.0, 4.0]),
            true,
        );
        let x2 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 0.0]),
            true,
        );
        let out = pairwise_distance(&x1, &x2, 2.0, 1e-6);
        assert_eq!(out.tensor.shape(), &[1]);
        // eps is added to each diff: norm of [3+eps, 4+eps].
        let eps = 1e-6;
        let expected = ((3.0 + eps).powi(2) + (4.0 + eps).powi(2)).sqrt();
        assert!((out.tensor.as_slice()[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn pairwise_distance_backward_matches_analytic_gradient() {
        // x1 = [[3, 4]], x2 = [[0, 0]], p = 2: d/dx1 = (x1 - x2 + eps)/||diff + eps||;
        // d/dx2 = -d/dx1. With eps = 1e-6 the shifted norm is 5.0000014.
        let eps = 1e-6;
        let x1 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[3.0, 4.0]),
            true,
        );
        let x2 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 0.0]),
            true,
        );
        let out = pairwise_distance(&x1, &x2, 2.0, eps);
        out.backward().expect("invariant: backward completes");
        let g1 = x1.grad().expect("x1 must receive a gradient");
        let g2 = x2.grad().expect("x2 must receive a gradient");
        let norm = ((3.0 + eps).powi(2) + (4.0 + eps).powi(2)).sqrt();
        let expected = [(3.0 + eps) / norm, (4.0 + eps) / norm];
        for (i, (&g, &e)) in g1.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-9,
                "pairwise_distance d/dx1[{i}]: got {g}, expected {e}"
            );
        }
        for (i, (&g, &e)) in g2.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g + e).abs() < 1e-9,
                "pairwise_distance d/dx2[{i}]: got {g}, expected {}",
                -e
            );
        }
    }

    #[test]
    fn pairwise_distance_p1_backward_matches_analytic() {
        // p = 1: out = sum|diff|; d/dx1 = sign(diff).
        let x1 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0, -2.0]),
            true,
        );
        let x2 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 0.0]),
            true,
        );
        let out = pairwise_distance(&x1, &x2, 1.0, 1e-6);
        out.backward().expect("invariant: backward completes");
        let g1 = x1.grad().expect("x1 must receive a gradient");
        let expected = [1.0, -1.0];
        for (i, (&g, &e)) in g1.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "pairwise_distance p=1 grad[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "2D")]
    fn pairwise_distance_rejects_non_2d() {
        let x1 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([2], &[1.0, 2.0]),
            true,
        );
        let x2 = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([2], &[0.0, 1.0]),
            true,
        );
        let _ = pairwise_distance(&x1, &x2, 2.0, 1e-6);
    }
}
