use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for margin ranking loss.
///
/// Computes `mean(max(0, -target * (input1 - input2) + margin))`.
/// The gradient w.r.t. `input1` is `-target / N * grad_out` and w.r.t.
/// `input2` is `target / N * grad_out`, both only where the hinge is active.
/// Forward and backward stay on the selected provider; the node retains the
/// provider-resident target tensor and hinge mask rather than host `Vec<T>`
/// payloads.
pub struct MarginRankingLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation (input1, input2).
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident target labels (+1 or -1).
    pub target_tensor: Tensor<T, B>,
    /// Provider-resident hinge activation mask (1.0 if active, 0.0 if not).
    pub mask: Tensor<T, B>,
    /// Number of elements in the loss reduction.
    pub n: usize,
    /// Provider-resident mean scale `1 / element_count`.
    pub mean_scale: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for MarginRankingLossNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "margin_ranking_loss"
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
        let scale = coeus_ops::mul(grad_out, &self.mean_scale, &backend);
        // d/d(input1) = -target * mask / N; d/d(input2) = +target * mask / N.
        let target_mask = coeus_ops::mul(&self.target_tensor, &self.mask, &backend);

        if let Some(Some(ref g1)) = input_grads.get(0) {
            let d1 = coeus_ops::mul(&coeus_ops::neg(&target_mask, &backend), &scale, &backend);
            coeus_ops::add_assign(g1.write(), &d1, &backend)?;
        }
        if let Some(Some(ref g2)) = input_grads.get(1) {
            let d2 = coeus_ops::mul(&target_mask, &scale, &backend);
            coeus_ops::add_assign(g2.write(), &d2, &backend)?;
        }
        Ok(())
    }
}

/// Tracked Margin Ranking loss (PyTorch `F.margin_ranking_loss` with
/// `reduction='mean'`).
///
/// `input1`, `input2`: shape `[N]`, `target`: `&[T]` of +1 or -1, `margin`:
/// threshold. Computes `mean(max(0, -target * (input1 - input2) + margin))`.
/// Returns a scalar Var (shape `[1]`). The complete forward and backward
/// computation stays on the selected provider; no input-sized host staging
/// occurs.
pub fn margin_ranking_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input1: &Var<T, B>,
    input2: &Var<T, B>,
    target: &[T],
    margin: T,
) -> Var<T, B> {
    let backend = B::default();
    let n = input1.tensor.numel();
    assert_eq!(
        input2.tensor.numel(),
        n,
        "input1 and input2 must have the same number of elements"
    );
    assert_eq!(target.len(), n, "target length must match input length");

    let shape = input1.tensor.shape_cloned();
    let target_tensor = Tensor::from_slice_on(shape.clone(), target, &backend);

    // hinge = relu(-target * (input1 - input2) + margin), all on-provider.
    let diff = coeus_ops::sub(&input1.tensor, &input2.tensor, &backend);
    let neg_target_diff =
        coeus_ops::mul(&coeus_ops::neg(&target_tensor, &backend), &diff, &backend);
    let raw = coeus_ops::add(
        &neg_target_diff,
        &Tensor::full_on(shape.clone(), margin, &backend),
        &backend,
    );
    let hinge = coeus_ops::relu(&raw, &backend);
    // mask = 1 where hinge > 0, else 0 (active hinge receives gradient).
    let zeros = Tensor::zeros_on(shape.clone(), &backend);
    let mask = coeus_ops::gt(&hinge, &zeros, &backend);
    let loss = coeus_ops::mean_axis(&hinge.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty margin-ranking reduction has axis zero");

    let requires_grad =
        crate::grad_mode::should_track_var(input1) || crate::grad_mode::should_track_var(input2);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = MarginRankingLossNode {
            output_grad,
            inputs: vec![input1.clone(), input2.clone()],
            target_tensor,
            mask,
            n,
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

    fn var_from(data: &[f64]) -> Var<f64, MoiraiBackend> {
        Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([data.len()], data),
            true,
        )
    }

    #[test]
    fn margin_ranking_forward_matches_reference() {
        // input1 = [1, 2, 3], input2 = [0, 2, 5], target = [1, -1, 1], margin = 1:
        //   hinge0 = relu(-1·(1-0) + 1) = relu(0) = 0
        //   hinge1 = relu(-(-1)·(2-2) + 1) = relu(1) = 1
        //   hinge2 = relu(-1·(3-5) + 1) = relu(3) = 3
        //   loss = (0 + 1 + 3) / 3
        let input1 = var_from(&[1.0, 2.0, 3.0]);
        let input2 = var_from(&[0.0, 2.0, 5.0]);
        let target = [1.0, -1.0, 1.0];
        let loss = margin_ranking_loss(&input1, &input2, &target, 1.0);
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - 4.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn margin_ranking_backward_matches_analytic_gradient() {
        // Active hinges at positions 1 and 2 (hinge > 0), inactive at 0.
        // d/dinput1 = -target·mask/n, d/dinput2 = +target·mask/n.
        let input1 = var_from(&[1.0, 2.0, 3.0]);
        let input2 = var_from(&[0.0, 2.0, 5.0]);
        let target = [1.0, -1.0, 1.0];
        let loss = margin_ranking_loss(&input1, &input2, &target, 1.0);
        loss.backward().expect("invariant: backward completes");
        let g1 = input1.grad().expect("input1 must receive a gradient");
        let g2 = input2.grad().expect("input2 must receive a gradient");
        // mask = [0, 1, 1]; d1 = -target·mask/3 = [0, 1/3, -1/3].
        let expected1 = [0.0, 1.0 / 3.0, -1.0 / 3.0];
        let expected2 = [0.0, -1.0 / 3.0, 1.0 / 3.0];
        for (i, (&g, &e)) in g1.as_slice().iter().zip(expected1.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "margin_ranking d1[{i}]: got {g}, expected {e}"
            );
        }
        for (i, (&g, &e)) in g2.as_slice().iter().zip(expected2.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "margin_ranking d2[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "target length must match input length")]
    fn margin_ranking_rejects_target_length_mismatch() {
        let input1 = var_from(&[1.0, 2.0]);
        let input2 = var_from(&[0.0, 1.0]);
        let target = [1.0];
        let _ = margin_ranking_loss(&input1, &input2, &target, 1.0);
    }
}
