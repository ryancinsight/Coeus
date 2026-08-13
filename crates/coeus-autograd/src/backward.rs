use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Reduce a broadcast gradient to match the original (smaller) input shape.
///
/// When an op broadcasts (e.g., `[3,1] + [1,4] → [3,4]`), the gradient
/// flowing back has shape `[3,4]` but the input's gradient should be `[3,1]`.
/// Broadcast axes are summed left-to-right, one [`coeus_ops::sum_axis`] call
/// per axis that actually needs reducing; a non-broadcast axis costs nothing.
///
/// The reduction predicate is evaluated against the running tensor rather than
/// precomputed into a side buffer, so the call allocates nothing of its own.
/// This is sound because `sum_axis` keeps the reduced axis at size 1: axis `d`
/// is only ever read before any reduction has touched index `d` or higher, so
/// its extent still equals the corresponding extent of the incoming gradient.
pub fn reduce_broadcast<
    T: Scalar,
    B: coeus_ops::ElementwiseOps<T> + coeus_ops::ReductionOps<T> + Default,
>(
    grad: Tensor<T, B>,
    target_shape: &[usize],
) -> Tensor<T, B> {
    let backend = B::default();
    let grad_ndim = grad.ndim();
    let target_ndim = target_shape.len();

    // Nothing to do if shapes already match.
    if grad.shape() == target_shape {
        return grad;
    }

    // Leading extra dims (grad has more dims than target): reduce axes 0..extra_dims.
    let extra_dims = grad_ndim.saturating_sub(target_ndim);

    let mut current = grad;

    // Sum out leading extra dims.  Each sum_axis collapses axis 0 and we must
    // then drop the kept-dim axis before proceeding, so we keep doing axis 0
    // while there are extra dims left (each sum reduces current ndim by 0 since
    // keep-dim is active; we reshape away the size-1 dim manually).
    for _ in 0..extra_dims {
        current = coeus_ops::sum_axis(&current, 0, &backend)
            .expect("invariant: broadcast reduction axis is validated");
        let new_shape = &current.shape()[1..];
        if new_shape.is_empty() {
            break;
        }
        current = current.reshape(new_shape);
    }

    // Sum out aligned broadcast dims. `sum_axis` keeps the reduced dimension at
    // size 1, so each target dimension still maps to the next axis position in
    // `current` after either branch.
    for axis in 0..target_ndim {
        if axis >= current.ndim() {
            break;
        }
        if target_shape[axis] == 1 && current.shape()[axis] > 1 {
            current = coeus_ops::sum_axis(&current, axis, &backend)
                .expect("invariant: broadcast reduction axis is validated");
        }
    }

    // ── Final reshape to exact target ────────────────────────────────────────
    if current.shape() != target_shape {
        current = current.reshape(target_shape);
    }

    current
}
