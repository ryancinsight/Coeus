use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for the soft-margin (logistic) loss.
///
/// Forward and backward stay on the selected provider: the node retains the
/// provider-resident margin `m = target * input` and the mean scale rather
/// than host `Vec<T>` payloads.
pub struct SoftMarginNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident margin `m = target * input`, saved for backward.
    pub margin: Tensor<T, B>,
    /// Number of elements in the mean reduction.
    pub n: usize,
    /// Original tensor shape for gradient reconstruction.
    pub shape: coeus_core::Shape,
    /// Provider-resident mean scale `1 / element_count`.
    pub mean_scale: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for SoftMarginNode<T, B> {
    fn op_name(&self) -> &'static str {
        "soft_margin"
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
        // d/d_input = -target * sigmoid(-m) / n
        // d/d_target = -input * sigmoid(-m) / n
        let scale = coeus_ops::mul(grad_out, &self.mean_scale, &backend);
        let neg_margin = coeus_ops::neg(&self.margin, &backend);
        let sig = coeus_ops::sigmoid(&neg_margin, &backend);
        let scaled_sig = coeus_ops::mul(&sig, &scale, &backend);

        if let Some(Some(ref g)) = input_grads.first() {
            let d_input = coeus_ops::mul(
                &coeus_ops::neg(&self.inputs[1].tensor, &backend),
                &scaled_sig,
                &backend,
            );
            coeus_ops::add_assign(g.write(), &d_input, &backend)?;
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let d_target = coeus_ops::mul(
                &coeus_ops::neg(&self.inputs[0].tensor, &backend),
                &scaled_sig,
                &backend,
            );
            coeus_ops::add_assign(g.write(), &d_target, &backend)?;
        }
        Ok(())
    }
}

/// Tracked soft-margin (logistic) loss, the `reduction="mean"` form of PyTorch
/// `SoftMarginLoss`: `mean_i log(1 + exp(-target_i * input_i))`, with `target`
/// in `{-1, +1}`. `input` and `target` must share shape.
///
/// Forward uses the provider `softplus(-m)` (stable for any margin) and
/// backward computes `-target * sigmoid(-m) / n` and `-input * sigmoid(-m) / n`
/// entirely on the selected provider.
pub fn soft_margin<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        input.tensor.shape(),
        target.tensor.shape(),
        "soft_margin requires input and target to have identical shapes"
    );
    let n = input.tensor.numel();
    assert!(n > 0, "soft_margin requires at least one element");
    let shape = input.tensor.shape_cloned();

    // m = target * input; loss = softplus(-m), all on-provider.
    let margin = coeus_ops::mul(&target.tensor, &input.tensor, &backend);
    let neg_margin = coeus_ops::neg(&margin, &backend);
    let per_elem = coeus_ops::softplus(&neg_margin, &backend);
    let loss = coeus_ops::mean_axis(&per_elem.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty soft-margin reduction has axis zero");

    let requires_grad =
        crate::grad_mode::should_track_var(input) || crate::grad_mode::should_track_var(target);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = SoftMarginNode {
            output_grad,
            inputs: vec![input.clone(), target.clone()],
            margin,
            n,
            shape,
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

    fn stable_softplus(x: f64) -> f64 {
        if x > 0.0 {
            x + (1.0 + (-x).exp()).ln()
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    #[test]
    fn soft_margin_forward_matches_reference() {
        // input = [1, -2, 0.5], target = [1, -1, 1]:
        //   m = [1, 2, 0.5]; loss = mean(softplus(-m)).
        let input = var_from(&[1.0, -2.0, 0.5]);
        let target = var_from(&[1.0, -1.0, 1.0]);
        let loss = soft_margin(&input, &target);
        let expected =
            (stable_softplus(-1.0) + stable_softplus(-2.0) + stable_softplus(-0.5)) / 3.0;
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn soft_margin_backward_matches_analytic_gradient() {
        // m = target*input; d/dinput = -target*sigmoid(-m)/n,
        // d/dtarget = -input*sigmoid(-m)/n.
        let input = var_from(&[1.0, -2.0, 0.5]);
        let target = var_from(&[1.0, -1.0, 1.0]);
        let loss = soft_margin(&input, &target);
        loss.backward().expect("invariant: backward completes");
        let input_grad = input.grad().expect("input must receive a gradient");
        let target_grad = target.grad().expect("target must receive a gradient");
        let sig = |m: f64| 1.0 / (1.0 + m.exp()); // sigmoid(-m) = 1/(1+e^m)
        for (i, (&g, (&x, &y))) in input_grad
            .as_slice()
            .iter()
            .zip(
                input
                    .tensor
                    .as_slice()
                    .iter()
                    .zip(target.tensor.as_slice().iter()),
            )
            .enumerate()
        {
            let m = x * y;
            let expected = -y * sig(m) / 3.0;
            assert!(
                (g - expected).abs() < 1e-12,
                "soft_margin input grad[{i}]: got {g}, expected {expected}"
            );
        }
        for (i, (&g, (&x, &y))) in target_grad
            .as_slice()
            .iter()
            .zip(
                input
                    .tensor
                    .as_slice()
                    .iter()
                    .zip(target.tensor.as_slice().iter()),
            )
            .enumerate()
        {
            let m = x * y;
            let expected = -x * sig(m) / 3.0;
            assert!(
                (g - expected).abs() < 1e-12,
                "soft_margin target grad[{i}]: got {g}, expected {expected}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "identical shapes")]
    fn soft_margin_rejects_shape_mismatch() {
        let input = var_from(&[1.0, 2.0]);
        let target = var_from(&[1.0]);
        let _ = soft_margin(&input, &target);
    }
}
