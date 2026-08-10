use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Backend bound required by the provider-resident Lp-norm forward/backward:
/// `BackendOps` provides elementwise (`abs`, `sign`, `mul`, `add_assign`) and
/// reduction (`sum_axis`) dispatch; `ScalarPowerOps` provides `pow_scalar`.
pub trait LpNormOps<T: Float>: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> {}
impl<T: Float, B> LpNormOps<T> for B where B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T>
{}

/// Autograd node for the global Lp norm `(Σ|xᵢ|^p)^(1/p)`.
///
/// The forward result is a provider-resident `[1]` tensor; backward keeps every
/// intermediate (`abs`, `sign`, powered magnitudes, the summed power, and the
/// norm) on the selected provider, with no input-sized host staging.
pub struct LpNormNode<T: Float, B: LpNormOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident `|xᵢ|` magnitudes.
    pub magnitudes: Tensor<T, B>,
    /// Provider-resident `sign(xᵢ)`.
    pub signs: Tensor<T, B>,
    /// Provider-resident `Σ|xᵢ|^p` (scalar tensor) — the forward summand.
    pub powered_sum: Tensor<T, B>,
    /// Provider-resident norm value `(Σ|xᵢ|^p)^(1/p)` as a `[1]` tensor.
    pub norm_value: Tensor<T, B>,
    /// The norm order `p` (as a scalar host value; `p == 1/p` is computed on
    /// the provider via scalar-power dispatch, not host loops).
    pub p: T,
    /// Number of elements in the reduction.
    pub n: usize,
    /// Original logical tensor shape.
    pub shape: coeus_core::Shape,
}

impl<T: Float, B: LpNormOps<T> + Default> BackwardNode<T, B> for LpNormNode<T, B> {
    fn op_name(&self) -> &'static str {
        "lp_norm"
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
        // d/dx ||x||_p = (x_i^(p-1)) * (||x||_p^(1-p)).
        // With x_i = sign(x_i)·|x_i|, and ||x||_p^(1-p) = (Σ|x|^p)^((1-p)/p),
        // the gradient factors through provider ops only:
        //   grad = sign(x) · |x|^(p-1) · norm^(1-p) · g_out
        // The |x|^(p-1) and norm^(1-p) terms are scalar powers of provider
        // tensors; the kink subgradient at x_i = 0 is zero (sign(0) = 0).
        let p_minus_one = self.p - T::one();
        let one_minus_p = T::one() - self.p;
        let powered_magnitudes = coeus_ops::pow_scalar(&self.magnitudes, p_minus_one, &backend);
        let scaled = coeus_ops::pow_scalar(&self.norm_value, one_minus_p, &backend);
        let gradient = coeus_ops::mul(&self.signs, &powered_magnitudes, &backend);
        let gradient = coeus_ops::mul(&gradient, &scaled, &backend);
        let gradient = coeus_ops::mul(&gradient, grad_out, &backend);

        if let Some(Some(ref gradient_buffer)) = input_grads.first() {
            coeus_ops::add_assign(gradient_buffer.write(), &gradient, &backend)?;
        }
        Ok(())
    }
}

/// Tracked global L2 norm: `sqrt(Σ xᵢ²)` over all elements.
///
/// Matches `torch.linalg.vector_norm(x, ord=2)` over a flattened view. All
/// arithmetic stays on the selected backend.
pub fn l2_norm<T: Float, B: LpNormOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    l_p_norm(input, T::from_f64(2.0))
}

/// Tracked global Lp norm: `(Σ|xᵢ|^p)^(1/p)` over all elements.
///
/// Matches `torch.linalg.vector_norm(x, ord=p)` over a flattened view for any
/// finite `p > 0`. The complete forward and backward computation stays on the
/// selected provider; no input-sized host staging occurs.
///
/// # Panics
/// Panics when the input is empty or `p` is not a finite positive value.
pub fn l_p_norm<T: Float, B: LpNormOps<T> + Default>(input: &Var<T, B>, p: T) -> Var<T, B> {
    let backend = B::default();
    let n = input.tensor.numel();
    assert!(n > 0, "lp_norm requires at least one element");
    assert!(
        p > T::zero() && <T as Float>::is_finite(p),
        "lp_norm: ord must be a finite positive number, got {p:?}"
    );

    let magnitudes = coeus_ops::abs(&input.tensor, &backend);
    let signs = coeus_ops::sign(&input.tensor, &backend);
    let powered = coeus_ops::pow_scalar(&magnitudes, p, &backend);
    let flattened = powered.reshape([n]);
    let powered_sum = coeus_ops::sum_axis(&flattened, 0, &backend)
        .expect("invariant: validated non-empty Lp reduction has axis zero");
    let norm_value = coeus_ops::pow_scalar(&powered_sum, T::one() / p, &backend);

    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad
            .as_ref()
            .expect("invariant: tracked output has a gradient buffer")
            .clone();
        let node = LpNormNode {
            output_grad,
            inputs: vec![input.clone()],
            magnitudes,
            signs,
            powered_sum,
            norm_value: norm_value.clone(),
            p,
            n,
            shape: input.tensor.shape_cloned(),
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var {
        tensor: norm_value,
        grad,
        creator,
    }
}

/// Autograd node for the per-axis Lp norm.
///
/// The reduced axis remains a size-one dimension in the output; backward
/// broadcasts the per-slice gradient back to the input shape through provider
/// broadcast multiply (no host staging).
pub struct LpNormAxisNode<T: Float, B: LpNormOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident `|xᵢ|` magnitudes (input shape).
    pub magnitudes: Tensor<T, B>,
    /// Provider-resident `sign(xᵢ)` (input shape).
    pub signs: Tensor<T, B>,
    /// Provider-resident per-slice `Σ|xᵢ|^p` (reduced shape, axis size 1).
    pub powered_sum: Tensor<T, B>,
    /// Provider-resident per-slice norm `(Σ|xᵢ|^p)^(1/p)` (reduced shape).
    pub norm_value: Tensor<T, B>,
    /// The norm order `p`.
    pub p: T,
    /// The reduced axis.
    pub axis: usize,
    /// Input logical shape.
    pub shape: coeus_core::Shape,
}

impl<T: Float, B: LpNormOps<T> + Default> BackwardNode<T, B> for LpNormAxisNode<T, B> {
    fn op_name(&self) -> &'static str {
        "lp_norm_axis"
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
        // Per-slice gradient: sign(x)·|x|^(p-1)·norm^(1-p)·g_out, with the
        // per-slice terms shaped for broadcast along `axis`.
        let p_minus_one = self.p - T::one();
        let one_minus_p = T::one() - self.p;
        let powered_magnitudes = coeus_ops::pow_scalar(&self.magnitudes, p_minus_one, &backend);
        let scaled = coeus_ops::pow_scalar(&self.norm_value, one_minus_p, &backend);
        let mut gradient = coeus_ops::mul(&self.signs, &powered_magnitudes, &backend);
        gradient = coeus_ops::mul(&gradient, &scaled, &backend);
        gradient = coeus_ops::mul(&gradient, grad_out, &backend);

        if let Some(Some(ref gradient_buffer)) = input_grads.first() {
            coeus_ops::add_assign(gradient_buffer.write(), &gradient, &backend)?;
        }
        Ok(())
    }
}

/// Tracked per-axis Lp norm: tensor reduced along `axis` to size 1, each slice
/// evaluated as `(Σ|xᵢ|^p)^(1/p)`.
///
/// Matches `torch.linalg.vector_norm(x, ord=p, dim=axis)`. The reduced axis
/// remains a size-one dimension in the output; forward and backward stay on
/// the selected provider.
///
/// # Panics
/// Panics when `axis` is out of range, the axis has zero elements, `p` is not
/// a finite positive value, or the input is empty.
pub fn l_p_norm_axis<T: Float, B: LpNormOps<T> + Default>(
    input: &Var<T, B>,
    p: T,
    axis: usize,
) -> Var<T, B> {
    let backend = B::default();
    assert!(
        axis < input.tensor.ndim(),
        "lp_norm_axis: axis {axis} out of bounds for rank {}",
        input.tensor.ndim()
    );
    let n_axis = input.tensor.shape()[axis];
    assert!(n_axis > 0, "lp_norm_axis: axis {axis} has zero elements");
    assert!(
        p > T::zero() && <T as Float>::is_finite(p),
        "lp_norm_axis: ord must be a finite positive number, got {p:?}"
    );

    let magnitudes = coeus_ops::abs(&input.tensor, &backend);
    let signs = coeus_ops::sign(&input.tensor, &backend);
    let powered = coeus_ops::pow_scalar(&magnitudes, p, &backend);
    let powered_sum = coeus_ops::sum_axis(&powered, axis, &backend)
        .expect("invariant: validated non-empty axis reduction");
    let norm_value = coeus_ops::pow_scalar(&powered_sum, T::one() / p, &backend);

    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            norm_value.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad
            .as_ref()
            .expect("invariant: tracked output has a gradient buffer")
            .clone();
        let node = LpNormAxisNode {
            output_grad,
            inputs: vec![input.clone()],
            magnitudes,
            signs,
            powered_sum,
            norm_value: norm_value.clone(),
            p,
            axis,
            shape: input.tensor.shape_cloned(),
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var {
        tensor: norm_value,
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
    fn l2_norm_forward_matches_reference() {
        let input = var_from(&[3.0, -4.0, 12.0]);
        let norm = l2_norm(&input);
        assert_eq!(norm.tensor.shape(), &[1]);
        // sqrt(9 + 16 + 144) = sqrt(169) = 13.
        assert!((norm.tensor.as_slice()[0] - 13.0).abs() < 1e-12);
    }

    #[test]
    fn lp_norm_forward_matches_reference_for_p1_and_p3() {
        let input = var_from(&[2.0, -3.0, 4.0]);
        // p = 1: |2| + |3| + |4| = 9.
        let p1 = l_p_norm(&input, 1.0);
        assert!((p1.tensor.as_slice()[0] - 9.0).abs() < 1e-12);
        // p = 3: (8 + 27 + 64)^(1/3) = 99^(1/3).
        let p3 = l_p_norm(&input, 3.0);
        let expected = 99.0_f64.powf(1.0 / 3.0);
        assert!((p3.tensor.as_slice()[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn lp_norm_backward_matches_analytic_gradient() {
        // x = [3, -4, 12], p = 2 → ||x|| = 13.
        // d/dx ||x||_2 = x / ||x||.
        let input = var_from(&[3.0, -4.0, 12.0]);
        let norm = l_p_norm(&input, 2.0);
        norm.backward().expect("invariant: backward completes");
        let grad = input.grad().expect("input must receive a gradient");
        let expected = [3.0 / 13.0, -4.0 / 13.0, 12.0 / 13.0];
        for (i, (&g, &e)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "lp_norm p=2 grad[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn lp_norm_p3_backward_matches_numeric_gradient() {
        // x = [2, -3, 4], p = 3, ||x||_3 = 99^(1/3).
        // d/dx = sign(x)·|x|^2·norm^(1-3).
        let input = var_from(&[2.0, -3.0, 4.0]);
        let norm = l_p_norm(&input, 3.0);
        norm.backward().expect("invariant: backward completes");
        let grad = input.grad().expect("input must receive a gradient");
        let norm_val = 99.0_f64.powf(1.0 / 3.0);
        let expected = [
            4.0 * norm_val.powf(-2.0),
            -9.0 * norm_val.powf(-2.0),
            16.0 * norm_val.powf(-2.0),
        ];
        for (i, (&g, &e)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-9,
                "lp_norm p=3 grad[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn lp_norm_axis_forward_and_backward_preserve_axis() {
        // Matrix [[1, 2], [3, 4]] with p = 2 along axis 0:
        // slice 0: sqrt(1 + 9) = sqrt(10); slice 1: sqrt(4 + 16) = sqrt(20).
        let base = Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let input = Var::new(base, true);
        let norm = l_p_norm_axis(&input, 2.0, 0);
        assert_eq!(norm.tensor.shape(), &[1, 2]);
        assert!((norm.tensor.as_slice()[0] - 10.0_f64.sqrt()).abs() < 1e-12);
        assert!((norm.tensor.as_slice()[1] - 20.0_f64.sqrt()).abs() < 1e-12);

        norm.backward().expect("invariant: axis backward completes");
        let grad = input.grad().expect("input must receive a gradient");
        // d/dx_ij = x_ij / slice_norm_j.
        let expected = [
            1.0 / 10.0_f64.sqrt(),
            2.0 / 20.0_f64.sqrt(),
            3.0 / 10.0_f64.sqrt(),
            4.0 / 20.0_f64.sqrt(),
        ];
        for (i, (&g, &e)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "lp_norm_axis grad[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "finite positive")]
    fn lp_norm_panics_on_zero_ord() {
        let input = var_from(&[1.0, 2.0]);
        let _ = l_p_norm(&input, 0.0);
    }

    #[test]
    #[should_panic(expected = "finite positive")]
    fn lp_norm_panics_on_non_finite_ord() {
        let input = var_from(&[1.0, 2.0]);
        let _ = l_p_norm(&input, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn lp_norm_axis_panics_on_bad_axis() {
        let input = var_from(&[1.0, 2.0]);
        let _ = l_p_norm_axis(&input, 2.0, 3);
    }
}
