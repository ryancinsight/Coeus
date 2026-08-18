use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for multi-label margin loss.
///
/// PyTorch `MultiLabelMarginLoss` with `reduction="mean"`:
/// x: `(N, C)`, target: `(N, C)` with `-1` = ignore padding.
/// Loss per sample: sum over valid targets `t` of sum over `j != t` of
/// `max(0, 1 - (x[t] - x[j]))`, normalized by `(N * C)`.
///
/// Forward and backward stay on the selected provider: the node retains the
/// provider-resident pairwise-active tensor `[N, C, C]` rather than host
/// `Vec<T>` payloads. The `target: &[isize]` host slice is a boundary upload;
/// the per-row target scores are gathered via the class-index boundary.
pub struct MultiLabelMarginLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Gradient buffer for the output scalar.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident pairwise-active mask `[N, C, C]` where `active[i,k,j]`
    /// is 1 when target position `k` holds a valid class, `j != target_val`,
    /// and `1 - x[i, target_val] + x[i, j] > 0`.
    pub active: Tensor<T, B>,
    /// Provider-resident one-hot of the target class values `[N, C]` (the
    /// `-1` padding mapped to 0), used to scatter the target-column gradient.
    pub target_onehot: Tensor<T, B>,
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

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.get(0) {
            // Each active (k, j) pair contributes +scale to x[j] and -scale to
            // x[target_val(k)]. incoming[j] = sum_k active[k, j] (the sibling
            // accumulation); outgoing[t] = sum over k of rowsum(k) scattered
            // to the target column via the one-hot of the target values.
            let scale = coeus_ops::mul(
                grad_out,
                &Tensor::full_on(
                    [1],
                    T::one() / T::from_f64((self.n * self.c) as f64),
                    &backend,
                ),
                &backend,
            );
            let incoming = coeus_ops::sum_axis(&self.active, 1, &backend)
                .expect("invariant: validated [N, C, C] active axis-1 reduction")
                .reshape([self.n, self.c]);
            // rowsum[i,k] = sum_j active[i,k,j]; scatter to target columns.
            let rowsum = coeus_ops::sum_axis(&self.active, 2, &backend)
                .expect("invariant: validated [N, C, C] active axis-2 reduction");
            // outgoing[i,t] = sum_k onehot[i,k,t] * rowsum[i,k]:
            // onehot is already [N, C, C] = (i, k, t); expand rowsum to match.
            let oh_kt = self.target_onehot.clone();
            let rowsum_kt = rowsum
                .reshape([self.n, self.c, 1])
                .broadcast([self.n, self.c, self.c])
                .to_contiguous_on(&backend);
            let scattered = coeus_ops::mul(&oh_kt, &rowsum_kt, &backend);
            let outgoing = coeus_ops::sum_axis(&scattered, 1, &backend)
                .expect("invariant: validated [N, C, C] outgoing axis-1 reduction")
                .reshape([self.n, self.c]);
            let per_elem = coeus_ops::sub(&incoming, &outgoing, &backend);
            let d_x = coeus_ops::mul(&per_elem, &scale, &backend);
            coeus_ops::add_assign(g.write(), &d_x, &backend)?;
        }
        Ok(())
    }
}

/// Tracked multi-label margin loss (PyTorch `MultiLabelMarginLoss` with
/// `reduction="mean"`).
///
/// `x`: shape `(N, C)`, target: `(N, C)` where `target[i][j] >= 0` are valid
/// class indices and `-1` means ignore. Returns a scalar `Var`.
///
/// The complete forward and backward computation stays on the selected
/// provider; no input-sized host staging occurs beyond the target boundary
/// upload (the per-row target scores are gathered with `gather`). The
/// pairwise formulation builds an `[N, C, C]` active tensor via broadcast.
pub fn multi_label_margin_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    target: &[isize],
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let shape = x.tensor.shape();
    assert_eq!(shape.len(), 2, "x must be 2D [batch_size, num_classes]");
    let n = shape[0];
    let c = shape[1];
    assert_eq!(target.len(), n * c, "target length must match N*C");

    let backend = B::default();

    // Target position k holds a class value (or -1 padding). Gather the
    // per-position target scores x[i, target_val] with a safe 0 sentinel for
    // the padding, and record which positions are valid.
    let mut safe_flat: Vec<T> = Vec::with_capacity(n * c);
    let mut valid_flat: Vec<T> = Vec::with_capacity(n * c);
    let mut target_flat: Vec<T> = Vec::with_capacity(n * c);
    for &v in target {
        valid_flat.push(if v >= 0 { T::one() } else { T::zero() });
        let safe = if v >= 0 { v as usize } else { 0 };
        target_flat.push(T::from_usize(safe));
        safe_flat.push(T::from_usize(safe));
    }
    let valid = Tensor::from_slice_on([n, c], &valid_flat, &backend);
    let safe_idx = Tensor::from_slice_on([n * c], &safe_flat, &backend);
    // Row i must take its own target columns, so this is a per-row gather with
    // an [N, C] index, not an `index_select` — the latter applies one column set
    // to every row and returns [N, N*C], which reshapes to [N, C] only when
    // N == 1.
    let safe_idx_rows = Tensor::from_slice_on([n, c], &safe_flat, &backend);
    let x_gathered = coeus_ops::gather(&x.tensor, 1, &safe_idx_rows, &backend);

    // Pairwise margin: m[i,k,j] = 1 - x[i, target_val(k)] + x[i, j] over all
    // target positions k and classes j. Materialize contiguous.
    let x_target_col = x_gathered.reshape([n, c, 1]);
    let x_row = x.tensor.reshape([n, 1, c]);
    let diff = coeus_ops::sub(&x_target_col, &x_row, &backend);
    let diff = diff.to_contiguous_on(&backend);
    let m = coeus_ops::add(
        &coeus_ops::neg(&diff, &backend),
        &Tensor::full_on([n, c, c], T::one(), &backend),
        &backend,
    );

    let hinge = coeus_ops::relu(&m, &backend);
    let active_margin = coeus_ops::gt(&hinge, &Tensor::zeros_on([n, c, c], &backend), &backend);
    // Valid target position k.
    let valid_k = valid.reshape([n, c, 1]);
    let valid_k_b = valid_k.broadcast([n, c, c]).to_contiguous_on(&backend);
    // j != target_val(k): one-hot of the target values over the flattened
    // positions [n*c, c], reshaped to [N, C, C], then negated (1 at j != t).
    let target_onehot = coeus_ops::one_hot(&safe_idx, c, &backend).reshape([n, c, c]);
    let same = target_onehot.reshape([n, c, c]).to_contiguous_on(&backend);
    let not_same = coeus_ops::where_cond(
        &same,
        &Tensor::zeros_on([n, c, c], &backend),
        &Tensor::full_on([n, c, c], T::one(), &backend),
        &backend,
    )
    .expect("multi_label_margin: j != target mask");
    // active = valid_k AND not_same AND active_margin.
    let active = coeus_ops::mul(&valid_k_b, &not_same, &backend);
    let active = coeus_ops::mul(&active, &active_margin, &backend);

    // loss = (1/(N*C)) * sum over all active pairs of hinge value.
    let weighted = coeus_ops::mul(&active, &hinge, &backend);
    let sum_c = coeus_ops::sum_axis(&weighted, 2, &backend)
        .expect("invariant: validated [N, C, C] active axis-2 reduction");
    let sum_t = coeus_ops::sum_axis(&sum_c, 1, &backend)
        .expect("invariant: validated [N, C] active axis-1 reduction");
    let loss_sum = coeus_ops::sum_axis(&sum_t, 0, &backend)
        .expect("invariant: validated [N] active axis-0 reduction");
    let loss = coeus_ops::mul(
        &loss_sum,
        &Tensor::full_on([1], T::one() / T::from_f64((n * c) as f64), &backend),
        &backend,
    );

    let out_tensor = loss.reshape([1]);
    let requires_grad = crate::grad_mode::should_track_var(x);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = MultiLabelMarginLossNode {
            output_grad,
            inputs: vec![x.clone()],
            active,
            target_onehot,
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
    fn multi_label_margin_forward_matches_reference() {
        // x = [[0.5, 0.8, -0.6]], target = [0, -1, -1] (one valid target):
        //   t_val=0, j=1: 1 - 0.5 + 0.8 = 1.3 > 0 (active)
        //   t_val=0, j=2: 1 - 0.5 - 0.6 = -0.1 (inactive)
        //   loss = 1.3 / (1*3) = 1.3/3.
        let x = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[0.5, 0.8, -0.6]),
            true,
        );
        let target = [0isize, -1, -1];
        let loss = multi_label_margin_loss(&x, &target);
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - 1.3 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn multi_label_margin_backward_matches_analytic() {
        // x = [[0.5, 0.8, -0.6]], target = [0, -1, -1]:
        //   active (k=0 → t_val=0, j=1) pair → dx[0] -= scale, dx[1] += scale.
        let x = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[0.5, 0.8, -0.6]),
            true,
        );
        let target = [0isize, -1, -1];
        let loss = multi_label_margin_loss(&x, &target);
        loss.backward().expect("invariant: backward completes");
        let g = x.grad().expect("x must receive a gradient");
        let expected = [-1.0 / 3.0, 1.0 / 3.0, 0.0];
        for (i, (&got, &e)) in g.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - e).abs() < 1e-12,
                "multi_label_margin grad[{i}]: got {got}, expected {e}"
            );
        }
    }

    #[test]
    fn multi_label_margin_two_valid_targets() {
        // x = [[0.5, 0.8, -0.6]], target = [0, 2, -1]:
        //   k=0 → t_val=0: j=1 active (1.3), j=2 inactive (-0.1)
        //   k=1 → t_val=2: j=0: 1 - (-0.6) + 0.5 = 2.1 active; j=1: 2.4 active
        //   loss = (1.3 + 2.1 + 2.4) / 3 = 5.8/3.
        let x = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[0.5, 0.8, -0.6]),
            true,
        );
        let target = [0isize, 2, -1];
        let loss = multi_label_margin_loss(&x, &target);
        assert!((loss.tensor.as_slice()[0] - 5.8 / 3.0).abs() < 1e-12);

        loss.backward().expect("invariant: backward completes");
        let g = x.grad().expect("x must receive a gradient");
        // Active pairs: (t=0,j=1), (t=2,j=0), (t=2,j=1) →
        //   dx[0] = -1/3 + 1/3 = 0; dx[1] = 1/3 + 1/3 = 2/3; dx[2] = -2/3.
        let expected = [0.0, 2.0 / 3.0, -2.0 / 3.0];
        for (i, (&got, &e)) in g.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - e).abs() < 1e-12,
                "multi_label_margin two-target grad[{i}]: got {got}, expected {e}"
            );
        }
    }

    #[test]
    fn multi_label_margin_forward_is_correct_for_a_batch() {
        // Regression: every other test of this op uses N = 1, the single batch
        // size at which its per-row target gather is shape-correct. At N = 2 the
        // op used to panic reshaping an [N, N*C] selection to [N, C].
        //
        // x = [[0.9, -0.3, 0.2, -1.1], [0.4, 1.5, -0.6, 0.75]],
        // targets: row 0 = {0}, row 1 = {1, 3}. Hinges 1 - (x[t] - x[j]) over
        // j != t, per this op's documented convention:
        //   row 0, t=0: [-0.2, 0.3, -1.0]   → 0.30
        //   row 1, t=1: [-0.1, -1.1, 0.25]  → 0.25
        //   row 1, t=3: [0.65, 1.75, -0.35] → 2.40
        //   total 2.95, normalized by N*C = 8 → 0.36875
        let x = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice(
                [2, 4],
                &[0.9, -0.3, 0.2, -1.1, 0.4, 1.5, -0.6, 0.75],
            ),
            true,
        );
        let target = [0isize, -1, -1, -1, 1, 3, -1, -1];
        let loss = multi_label_margin_loss(&x, &target);
        assert_eq!(loss.tensor.shape(), &[1]);
        let expected = 0.368_75;
        let actual = loss.tensor.as_slice()[0];
        assert!(
            (actual - expected).abs() < 1e-12,
            "batched multi_label_margin: got {actual}, expected {expected}"
        );
    }

    #[test]
    #[should_panic(expected = "target length must match N*C")]
    fn multi_label_margin_rejects_target_length_mismatch() {
        let x = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[0.5, 0.8, -0.6]),
            true,
        );
        let target = [0isize];
        let _ = multi_label_margin_loss(&x, &target);
    }
}
