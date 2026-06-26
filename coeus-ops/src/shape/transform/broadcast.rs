// ── broadcast_to — materialize broadcasted tensor ──
//
// Repeats elements along singleton dimensions without changing rank.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Materialize `input` into `target_shape` by repeating along dimensions
/// whose source extent is `1`.
///
/// This follows standard NumPy/PyTorch broadcasting rules, except that the
/// rank must already match (`target_shape.len() == input.ndim()`).
///
/// # Panics
/// Panics if `target_shape.len() != input.ndim()` or if any dimension is
/// incompatible for broadcasting.
#[inline]
pub fn broadcast_to<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    target_shape: &[usize],
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        target_shape.len(),
        input.ndim(),
        "broadcast_to: target rank {} must match input rank {}",
        target_shape.len(),
        input.ndim(),
    );

    for (dim, (&src, &dst)) in input.shape().iter().zip(target_shape.iter()).enumerate() {
        assert!(
            src == dst || src == 1,
            "broadcast_to: incompatible dimension at axis {dim}: input={src}, target={dst}"
        );
    }

    let numel: usize = target_shape.iter().product();
    let out_vec: Vec<T> = (0..numel)
        .map(|flat| {
            let out_idx = crate::shape::flat_to_nd(flat, target_shape);
            let src_idx: Vec<usize> = out_idx
                .iter()
                .enumerate()
                .map(|(dim, &idx)| if input.shape()[dim] == 1 { 0 } else { idx })
                .collect();
            input.get(&src_idx)
        })
        .collect();

    Tensor::from_slice_on(target_shape.to_vec(), &out_vec, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn broadcast_1d_repeats_singleton_value() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![1], &[3.5f32]);
        let out = broadcast_to(&input, &[4], &b);
        assert_eq!(out.shape(), &[4]);
        assert_eq!(out.as_slice(), &[3.5, 3.5, 3.5, 3.5]);
    }

    #[test]
    fn broadcast_2d_repeats_along_first_axis() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![1, 3], &[1.0f32, 2.0, 3.0]);
        let out = broadcast_to(&input, &[2, 3], &b);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn broadcast_identity_returns_same_values() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0]);
        let out = broadcast_to(&input, &[2, 2], &b);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.as_slice(), input.as_slice());
    }

    #[test]
    #[should_panic(expected = "broadcast_to")]
    fn broadcast_incompatible_shape_panics() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![2], &[1.0f32, 2.0]);
        let _ = broadcast_to(&input, &[3], &b);
    }
}

