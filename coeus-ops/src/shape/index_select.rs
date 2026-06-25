// ── index_select — select slices from a tensor along one axis ──
//
// `index_select(input, dim, index)` returns a tensor with the same number of
// dimensions as `input` where the `dim`-th axis is replaced by the selected
// slices at positions given by the 1-D `index` tensor.
//
// This is the exact semantic of `torch.index_select(input, dim, index)`.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Select slices from `input` along `dim` at positions given by `index`.
///
/// `index` must be 1-D. The output shape equals `input.shape()` with
/// `dim` replaced by `index.len()`.
///
/// # Panics
/// - `index` must be 1-D.
/// - `dim` must be < `input.ndim()`.
/// - Every value in `index` must be in `[0, input.shape()[dim])`.
#[inline]
pub fn index_select<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    dim: usize,
    index: &Tensor<T, B>,
    _backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    assert!(
        dim < ndim,
        "index_select: dim {dim} out of range for {ndim}-D tensor"
    );
    assert_eq!(
        index.ndim(),
        1,
        "index_select: index must be 1-D, got {}-D",
        index.ndim()
    );

    let in_shape = input.shape();
    let k = index.shape()[0]; // number of selected slices

    // Build output shape: replace in_shape[dim] with k.
    let mut out_shape = in_shape.to_vec();
    out_shape[dim] = k;

    // Materialise contiguous views.
    let in_cont = input.to_contiguous();
    let idx_cont = index.to_contiguous();
    let in_s = in_cont.as_slice();
    let idx_s = idx_cont.as_slice();

    // Compute row-major strides for the input.
    let mut in_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }

    let out_numel: usize = out_shape.iter().product();
    let mut out_data = vec![T::zero(); out_numel];

    // Compute row-major strides for the output.
    let mut out_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }

    for out_flat in 0..out_numel {
        // Decode output flat → multi-dim.
        let mut coords = vec![0usize; ndim];
        let mut rem = out_flat;
        for d in 0..ndim {
            coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }

        // The `dim`-th coordinate indexes into `index`, giving us the
        // source position in the input along `dim`.
        let sel = idx_s[coords[dim]].to_f64() as usize;
        assert!(
            sel < in_shape[dim],
            "index_select: index value {sel} out of bounds for dim {dim} size {}",
            in_shape[dim]
        );

        // Compute input flat offset, substituting `sel` at `dim`.
        let mut in_flat = 0usize;
        for d in 0..ndim {
            let c = if d == dim { sel } else { coords[d] };
            in_flat += c * in_strides[d];
        }
        out_data[out_flat] = in_s[in_flat];
    }

    Tensor::from_slice(out_shape, &out_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn index_select_1d_selects_correct_elements() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![5], &[10.0f32, 20.0, 30.0, 40.0, 50.0]);
        let idx = Tensor::from_slice(vec![3], &[4.0f32, 0.0, 2.0]);
        let out = index_select(&x, 0, &idx, &b);
        assert_eq!(out.shape(), &[3]);
        assert_eq!(out.as_slice(), &[50.0, 10.0, 30.0]);
    }

    #[test]
    fn index_select_2d_selects_rows() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(
            vec![4, 3],
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let idx = Tensor::from_slice(vec![2], &[3.0f32, 1.0]);
        let out = index_select(&x, 0, &idx, &b);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[10.0, 11.0, 12.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn index_select_2d_selects_cols() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![2, 4], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let idx = Tensor::from_slice(vec![2], &[3.0f32, 0.0]);
        let out = index_select(&x, 1, &idx, &b);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.as_slice(), &[4.0, 1.0, 8.0, 5.0]);
    }
}
