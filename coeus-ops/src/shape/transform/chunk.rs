// -- chunk --
// Split a tensor into N approximately equal-size chunks along a dimension.

use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Split `x` into at most `chunks` pieces along `dim`.
///
/// Each piece has size `ceil(dim_size / chunks)`. The last may be smaller.
/// Equivalent to `torch.chunk(input, chunks, dim)`.
#[inline]
pub fn chunk<T: Scalar, B: coeus_core::ComputeBackend + Default>(
    x: &Tensor<T, B>,
    chunks: usize,
    dim: usize,
) -> Vec<Tensor<T, B>>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = x.ndim();
    assert!(
        dim < ndim,
        "chunk: dim {dim} out of range for {ndim}D tensor"
    );
    assert!(chunks > 0, "chunk: chunks must be greater than zero");
    let dim_size = x.shape()[dim];
    if dim_size == 0 {
        return vec![];
    }
    let chunk_size = dim_size.div_ceil(chunks);
    crate::shape::concat_split_stack::split(x, chunk_size, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn chunk_1d_even_split() {
        let x =
            Tensor::<f32, SequentialBackend>::from_slice(vec![6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let parts = chunk(&x, 3, 0);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]);
        assert_eq!(parts[0].as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn chunk_1d_uneven_last() {
        let x = Tensor::<f32, SequentialBackend>::from_slice(vec![5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let parts = chunk(&x, 3, 0);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[2].shape(), &[1]);
        assert_eq!(parts[2].as_slice(), &[5.0]);
    }
}

