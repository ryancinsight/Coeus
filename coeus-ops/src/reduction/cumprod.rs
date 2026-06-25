// ── cumprod — cumulative product along a dimension ──
//
// Matches `torch.cumprod(input, dim)`.

use crate::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Inclusive cumulative product of `x` along `dim`.
///
/// Output has the same shape as `x`.  `out[i] = x[0] * x[1] * … * x[i]`
/// along the specified dimension.
///
/// # Panics
/// Panics if `dim >= x.ndim()`.
#[inline]
pub fn cumprod<T: Scalar, B: BackendOps<T> + Default>(
    x: &Tensor<T, B>,
    dim: usize,
    _backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = x.ndim();
    assert!(
        dim < ndim,
        "cumprod: dim {dim} out of range for {ndim}-D tensor"
    );
    let shape = x.shape();
    let numel: usize = shape.iter().product();
    let n = shape[dim]; // length of the product dimension

    // Compute row-major strides.
    let mut strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    let dim_stride = strides[dim];

    let x_cont = x.to_contiguous();
    let x_s = x_cont.as_slice();
    let mut out = vec![T::zero(); numel];

    // Iterate over all "lines" along dim.
    // A line is identified by all coordinates except the dim axis.
    let outer = numel / (n * dim_stride); // number of outer slices
    let inner = dim_stride; // spacing between adjacent elements along dim

    for outer_idx in 0..outer {
        for inner_idx in 0..inner {
            let base = outer_idx * n * inner + inner_idx;
            let mut acc = T::one();
            for i in 0..n {
                let flat = base + i * inner;
                acc = acc * x_s[flat];
                out[flat] = acc;
            }
        }
    }

    Tensor::from_slice(shape.to_vec(), &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn cumprod_1d_matches_prefix_products() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]);
        let out = cumprod(&x, 0, &b);
        assert_eq!(out.shape(), &[4]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn cumprod_2d_along_axis0() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![3, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // dim=0: col0 = 1,3,15  col1 = 2,8,48
        let out = cumprod(&x, 0, &b);
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 8.0, 15.0, 48.0]);
    }

    #[test]
    fn cumprod_2d_along_axis1() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // dim=1: row0=[1,2,6]  row1=[4,20,120]
        let out = cumprod(&x, 1, &b);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);
    }

    #[test]
    fn cumprod_ones_is_identity() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![5], &[1.0f32; 5]);
        let out = cumprod(&x, 0, &b);
        assert_eq!(out.as_slice(), &[1.0; 5]);
    }
}
