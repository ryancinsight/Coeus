use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Reduce a broadcast gradient to match the original (smaller) input shape.
///
/// When an op broadcasts (e.g., `[3,1] + [1,4] → [3,4]`), the gradient
/// flowing back has shape `[3,4]` but the input's gradient should be `[3,1]`.
/// This function sums over the broadcast dimensions.
pub fn reduce_broadcast<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    grad: Tensor<T, B>,
    target_shape: &[usize],
) -> Tensor<T, B> {
    let mut current = grad;
    let backend = B::default();

    // Sum out leading extra dims
    while current.ndim() > target_shape.len() {
        current = coeus_ops::sum_axis(&current, 0, &backend);
        let new_shape = &current.shape()[1..];
        if new_shape.is_empty() {
            break;
        }
        current = current.reshape(new_shape);
    }

    // Sum out broadcast dims (where target is 1 but current is > 1)
    for i in 0..target_shape.len() {
        if i < current.ndim() && target_shape[i] == 1 && current.shape()[i] > 1 {
            current = coeus_ops::sum_axis(&current, i, &backend);
        }
    }

    // Reshape to exact target
    if current.shape() != target_shape {
        current = current.reshape(target_shape);
    }

    current
}
