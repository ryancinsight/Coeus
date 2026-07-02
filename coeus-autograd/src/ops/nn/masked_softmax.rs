use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::ops::nn::softmax::accumulate_softmax_grad;
use crate::var::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for masked and causal softmax.
///
/// Stores the (masked) softmax output `y`. The reverse pass is the ordinary softmax
/// jacobian applied to `y` (see `accumulate_softmax_grad`): masked positions hold
/// `y = 0`, so they receive zero gradient and do not enter the per-row sum, and an
/// all-masked row propagates zero — no `-inf`/`NaN` is ever formed.
pub struct MaskedSoftmaxNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for this node's output.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Tracked input (the pre-softmax scores).
    pub inputs: Vec<Var<T, B>>,
    /// Saved masked-softmax output `y` for backward.
    pub y_clone: Tensor<T, B>,
    /// Axis the softmax was reduced over.
    pub dim_u: usize,
    /// Trace label (`"masked_softmax"` / `"causal_softmax"`).
    pub op: &'static str,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for MaskedSoftmaxNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        self.op
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
        if let Some(Some(ref g_in)) = input_grads.get(0) {
            accumulate_softmax_grad(grad_out, &self.y_clone, self.dim_u, g_in);
        }
    }
}

fn normalize_dim(dim: isize, ndim: usize, op: &str) -> usize {
    let dim_u = if dim < 0 { ndim as isize + dim } else { dim };
    assert!(
        dim_u >= 0 && (dim_u as usize) < ndim,
        "{op}: dim {dim} out of bounds for ndim={ndim}"
    );
    dim_u as usize
}

fn build_var<T, B>(
    input: &Var<T, B>,
    y_t: Tensor<T, B>,
    dim_u: usize,
    op: &'static str,
) -> Var<T, B>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
{
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            y_t.shape_cloned(),
            &backend,
        )))
    });
    let creator = grad.as_ref().map(|g| {
        Arc::new(MaskedSoftmaxNode {
            output_grad: g.clone(),
            inputs: vec![input.clone()],
            y_clone: y_t.clone(),
            dim_u,
            op,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: y_t,
        grad,
        creator,
    }
}

/// Tracked masked softmax over `dim`.
///
/// Computes softmax across positions where `mask != 0`; masked positions (and any
/// fully-masked row) are zero. Gradient flows to `input` only — `mask` is data.
///
/// # Panics
/// If `dim` is out of range or `input`/`mask` shapes differ.
#[must_use]
pub fn masked_softmax<T, B>(input: &Var<T, B>, mask: &Tensor<T, B>, dim: isize) -> Var<T, B>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let dim_u = normalize_dim(dim, input.tensor.ndim(), "masked_softmax");
    let backend = B::default();
    let y_t = coeus_ops::masked_softmax(&input.tensor, mask, dim_u, &backend);
    build_var(input, y_t, dim_u, "masked_softmax")
}

/// Tracked causal (lower-triangular) softmax over `dim`: future positions are masked
/// before softmax. Gradient flows to `input`.
///
/// # Panics
/// If `dim` is out of range.
#[must_use]
pub fn causal_softmax<T, B>(input: &Var<T, B>, dim: isize) -> Var<T, B>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let dim_u = normalize_dim(dim, input.tensor.ndim(), "causal_softmax");
    let backend = B::default();
    let y_t = coeus_ops::causal_softmax(&input.tensor, dim_u, &backend);
    build_var(input, y_t, dim_u, "causal_softmax")
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;

    #[test]
    fn masked_softmax_forward_and_gradient() {
        // input=[[1,2,3]], mask=[[1,1,0]], dim=1: softmax over cols 0,1; col 2 -> 0.
        let input =
            Var::<f64, MoiraiBackend>::new(Tensor::from_slice([1, 3], &[1.0, 2.0, 3.0]), true);
        let mask = Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[1.0, 1.0, 0.0]);
        let out = masked_softmax(&input, &mask, 1);
        let (e1, e2) = (1.0_f64.exp(), 2.0_f64.exp());
        let (y0, y1) = (e1 / (e1 + e2), e2 / (e1 + e2));
        let y = out.tensor.as_slice();
        assert!((y[0] - y0).abs() < 1e-12);
        assert!((y[1] - y1).abs() < 1e-12);
        assert!(
            y[2].abs() < 1e-12,
            "masked position must be 0, got {}",
            y[2]
        );

        // Seed grad_out=[1,0,0]: dx_k = y_k*(g_k - sum_j y_j g_j), sum = y0.
        out.backward_with_seed(Tensor::from_slice([1, 3], &[1.0, 0.0, 0.0]));
        let g = input.grad().unwrap();
        let gs = g.as_slice();
        assert!((gs[0] - y0 * (1.0 - y0)).abs() < 1e-12, "dx0: {}", gs[0]);
        assert!((gs[1] - y1 * (-y0)).abs() < 1e-12, "dx1: {}", gs[1]);
        assert!(
            gs[2].abs() < 1e-12,
            "masked input grad must be 0, got {}",
            gs[2]
        );
    }

    #[test]
    fn masked_softmax_all_masked_row_is_zero_no_nan() {
        // A fully-masked row must yield all-zero output and a finite all-zero gradient.
        let input =
            Var::<f64, MoiraiBackend>::new(Tensor::from_slice([1, 3], &[1.0, 2.0, 3.0]), true);
        let mask = Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[0.0, 0.0, 0.0]);
        let out = masked_softmax(&input, &mask, 1);
        for &v in out.tensor.as_slice() {
            assert_eq!(v, 0.0, "all-masked output must be 0");
        }
        out.backward();
        for &v in input.grad().unwrap().as_slice() {
            assert!(
                v.is_finite() && v == 0.0,
                "all-masked grad must be finite 0, got {v}"
            );
        }
    }

    #[test]
    fn causal_softmax_is_lower_triangular_and_differentiable() {
        // 2x2: row0 attends only to col0 (future masked); row1 to cols 0,1.
        let input =
            Var::<f64, MoiraiBackend>::new(Tensor::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]), true);
        let out = causal_softmax(&input, 1);
        let y = out.tensor.as_slice();
        assert!((y[0] - 1.0).abs() < 1e-12, "causal row0 col0");
        assert!(y[1].abs() < 1e-12, "causal row0 col1 must be masked (0)");
        let (e3, e4) = (3.0_f64.exp(), 4.0_f64.exp());
        assert!((y[2] - e3 / (e3 + e4)).abs() < 1e-12, "causal row1 col0");
        assert!((y[3] - e4 / (e3 + e4)).abs() < 1e-12, "causal row1 col1");

        out.backward();
        assert!(input
            .grad()
            .unwrap()
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
    }
}
