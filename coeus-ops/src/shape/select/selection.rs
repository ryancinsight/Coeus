// -- one_hot / masked_select --

use crate::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, Scalar};
use coeus_tensor::Tensor;

/// One-hot encoding: integer indices to float indicator matrix.
///
/// `indices` is a 1-D tensor of non-negative integer values (stored as `T`).
/// Returns `[n, num_classes]` with `1.0` at column `indices[i]`.
#[inline]
pub fn one_hot<T: Scalar, B: BackendOps<T> + Default>(
    indices: &Tensor<T, B>,
    num_classes: usize,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        indices.ndim(),
        1,
        "one_hot: indices must be 1-D, got {}-D",
        indices.ndim()
    );
    let n = indices.shape()[0];
    let idx_cont = indices.to_contiguous();
    let idx_slice = idx_cont.as_slice();
    let mut data = vec![T::zero(); n * num_classes];
    for (row, &v) in idx_slice.iter().enumerate() {
        let idx = v.to_f64();
        assert!(
            idx.is_finite() && idx >= 0.0 && idx.fract() == 0.0,
            "one_hot: index value {idx} is not a non-negative integer"
        );
        let col = idx as usize;
        assert!(
            col < num_classes,
            "one_hot: index {col} out of range for num_classes={num_classes}"
        );
        data[row * num_classes + col] = T::one();
    }
    Tensor::from_slice_on(vec![n, num_classes], &data, backend)
}

/// Select elements of `input` where `mask` is non-zero. Returns 1-D tensor.
#[inline]
pub fn masked_select<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    mask: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        input.shape(),
        mask.shape(),
        "masked_select: input shape {:?} must match mask {:?}",
        input.shape(),
        mask.shape()
    );
    let in_cont = input.to_contiguous();
    let m_cont = mask.to_contiguous();
    let selected: Vec<T> = in_cont
        .as_slice()
        .iter()
        .zip(m_cont.as_slice().iter())
        .filter_map(|(&v, &m)| if m != T::zero() { Some(v) } else { None })
        .collect();
    let len = selected.len();
    Tensor::from_slice_on(vec![len], &selected, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn one_hot_basic() {
        let b = SequentialBackend::new();
        let idx = Tensor::<f32, SequentialBackend>::from_slice(vec![4], &[0.0, 2.0, 1.0, 2.0]);
        let oh = one_hot(&idx, 3, &b);
        assert_eq!(oh.shape(), &[4, 3]);
        assert_eq!(
            oh.as_slice(),
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn masked_select_basic() {
        let b = SequentialBackend::new();
        let x = Tensor::<f32, SequentialBackend>::from_slice(vec![4], &[1.0, 2.0, 3.0, 4.0]);
        let m = Tensor::<f32, SequentialBackend>::from_slice(vec![4], &[0.0, 1.0, 0.0, 1.0]);
        let out = masked_select(&x, &m, &b);
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.as_slice(), &[2.0, 4.0]);
    }
}
