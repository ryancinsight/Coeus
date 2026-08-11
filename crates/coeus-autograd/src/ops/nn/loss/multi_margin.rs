use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for the multi-class margin loss.
///
/// Forward and backward stay on the selected provider: the node retains the
/// provider-resident unit gradient `coef` tensor and target mask rather than
/// host `Vec<T>` payloads.
pub struct MultiMarginNode<
    T: Float,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident `p * relu(m)^(p-1) / (N*C)` per element, `[N, C]`.
    pub grad_unit: Tensor<T, B>,
    /// Provider-resident one-hot target mask, `[N, C]`.
    pub target_mask: Tensor<T, B>,
    /// Batch size.
    pub n: usize,
    /// Number of classes.
    pub c: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default>
    BackwardNode<T, B> for MultiMarginNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "multi_margin"
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
        if let Some(Some(ref g)) = input_grads.first() {
            // d/dx_ij = grad_unit_ij - mask_ij * rowsum(grad_unit)
            // (the target column receives the negation of every active
            // sibling coefficient). All on-provider.
            let rowsum = coeus_ops::sum_axis(&self.grad_unit, 1, &backend)
                .expect("invariant: validated [N, C] multi-margin row reduction");
            let target_term = coeus_ops::mul(
                &self.target_mask,
                &rowsum.broadcast([self.n, self.c]),
                &backend,
            );
            let d_x = coeus_ops::sub(&self.grad_unit, &target_term, &backend);
            let d_x = coeus_ops::mul(&d_x, &grad_out.broadcast([self.n, self.c]), &backend);
            coeus_ops::add_assign(g.write(), &d_x, &backend)?;
        }
        Ok(())
    }
}

/// Tracked multi-class margin loss (PyTorch `MultiMarginLoss`, `reduction="mean"`):
/// `mean_i (1/C) sum_{j != y_i} max(0, margin - x[i,y_i] + x[i,j])^p`.
///
/// `x`: `[N, C]` scores, `targets`: `[N]` class indices, `p >= 1`, `margin`.
/// Gradient (unit upstream): for `j != y_i` with `m_ij = margin - x[i,y_i] + x[i,j] > 0`,
/// `d/d x_ij = p*m_ij^(p-1) / (N*C)`; the target column accumulates the negation of
/// every active sibling term.
///
/// The complete forward and backward computation stays on the selected
/// provider; the `targets: &[usize]` host slice is a one-hot boundary upload.
pub fn multi_margin<
    T: Float,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
>(
    x: &Var<T, B>,
    targets: &[usize],
    p: T,
    margin: T,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let shape = x.tensor.shape();
    assert_eq!(shape.len(), 2, "multi_margin expects 2D [N, C] input");
    let n = shape[0];
    let c = shape[1];
    assert_eq!(targets.len(), n, "targets length must match batch size");

    // x[·, y] via index-select; m = margin - x[i,y] + x[i,j] (all j).
    let target_f: Vec<T> = targets.iter().map(|&i| T::from_usize(i)).collect();
    let target_tensor = Tensor::from_slice_on([n], &target_f, &backend);
    let x_target = coeus_ops::index_select(&x.tensor, 1, &target_tensor, &backend).reshape([n, 1]);
    let margin_tensor = Tensor::full_on([n, 1], margin, &backend);
    let m = coeus_ops::add(
        &coeus_ops::sub(&margin_tensor, &x_target, &backend),
        &x.tensor,
        &backend,
    );
    let hinge = coeus_ops::relu(&m, &backend);

    // loss = sum over all j of hinge^p, minus the j == y term (hinge[y] =
    // margin, a constant that must be excluded). All on-provider.
    let powered = coeus_ops::pow_scalar(&hinge, p, &backend);
    let row_sum = coeus_ops::sum_axis(&powered, 1, &backend)
        .expect("invariant: validated [N, C] multi-margin row reduction");
    let margin_p = coeus_ops::pow_scalar(&margin_tensor, p, &backend);
    let row_net = coeus_ops::sub(&row_sum, &margin_p.reshape([n]), &backend);
    // loss = sum_i row_net_i / (N * C). `mean_axis` divides by N, so scale
    // the mean by 1/C.
    let inv_c = T::one() / T::from_f64(c as f64);
    let mean_loss = coeus_ops::mean_axis(&row_net.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty multi-margin reduction has axis zero");
    let loss = coeus_ops::mul(&mean_loss, &Tensor::full_on([1], inv_c, &backend), &backend);

    // grad_unit = p * relu(m)^(p-1) / (N*C), all j including j == y, but
    // zeroed where the hinge is inactive (m <= 0) so `0^(p-1)` (which is 1
    // for p == 1) never activates a dead sibling.
    let p_minus_one = p - T::one();
    let raw_coef = coeus_ops::mul(
        &coeus_ops::pow_scalar(&hinge, p_minus_one, &backend),
        &Tensor::full_on(shape.to_vec(), p, &backend),
        &backend,
    );
    let active = coeus_ops::gt(&m, &Tensor::zeros_on(shape.to_vec(), &backend), &backend);
    let coef = coeus_ops::where_cond(
        &active,
        &raw_coef,
        &Tensor::zeros_on(shape.to_vec(), &backend),
        &backend,
    )
    .expect("multi_margin: active-hinge mask");
    let inv_nc = T::one() / T::from_f64((n * c) as f64);
    let grad_unit = coeus_ops::mul(
        &coef,
        &Tensor::full_on(shape.to_vec(), inv_nc, &backend),
        &backend,
    );
    let target_mask = coeus_ops::one_hot(&target_tensor, c, &backend);

    let out_tensor = loss.reshape([1]);
    let requires_grad = crate::grad_mode::should_track_var(x);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = MultiMarginNode {
            output_grad,
            inputs: vec![x.clone()],
            grad_unit,
            target_mask,
            n,
            c,
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
    fn multi_margin_forward_matches_reference() {
        // x = [[1, 2, 3]], target = [0], p = 1, margin = 1:
        //   m_j = 1 - x[0] + x[j] = [1, 2, 3]; hinge relu = [1, 2, 3].
        //   sum_j≠0 = 2 + 3 = 5; loss = 5 / (1*3) = 5/3.
        let x = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[1.0, 2.0, 3.0]),
            true,
        );
        let targets = [0usize];
        let loss = multi_margin(&x, &targets, 1.0, 1.0);
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - 5.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn multi_margin_backward_matches_analytic() {
        // x = [[1, 2, 3]], target = [0], p = 1, margin = 1:
        //   coef_j = relu(m)^0 = 1 for all j; grad_unit = 1/3.
        //   d/dx_j = grad_unit - mask_j * rowsum = [1/3 - 1, 1/3, 1/3].
        let x = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[1.0, 2.0, 3.0]),
            true,
        );
        let targets = [0usize];
        let loss = multi_margin(&x, &targets, 1.0, 1.0);
        loss.backward().expect("invariant: backward completes");
        let grad = x.grad().expect("x must receive a gradient");
        let expected = [1.0 / 3.0 - 1.0, 1.0 / 3.0, 1.0 / 3.0];
        for (i, (&g, &e)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "multi_margin grad[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn multi_margin_p2_backward_matches_numeric() {
        // x = [[0, 1]], target = [1], p = 2, margin = 0.5:
        //   m_j = 0.5 - x[1] + x[j] = [-0.5, 0.5]; hinge = [0, 0.5].
        //   coef = p*hinge^(p-1) = [0, 1]; grad_unit = [0, 1/2].
        //   rowsum = 0.5; d/dx = [0 - 0, 0.5 - 0.5] = [0, 0].
        let x = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 1.0]),
            true,
        );
        let targets = [1usize];
        let loss = multi_margin(&x, &targets, 2.0, 0.5);
        loss.backward().expect("invariant: backward completes");
        let grad = x.grad().expect("x must receive a gradient");
        for (i, &g) in grad.as_slice().iter().enumerate() {
            assert!((g - 0.0).abs() < 1e-12, "multi_margin p=2 grad[{i}] = {g}");
        }
    }

    #[test]
    #[should_panic(expected = "targets length must match batch size")]
    fn multi_margin_rejects_target_mismatch() {
        let x = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[1.0, 2.0, 3.0]),
            true,
        );
        let targets = [0usize, 1usize];
        let _ = multi_margin(&x, &targets, 1.0, 1.0);
    }
}
