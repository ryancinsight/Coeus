// ── Tensor views ──
// Zero-copy slice, transpose, reshape, and permute operations.
// All share the underlying storage — no data copy.

use std::marker::PhantomData;

use crate::tensor::Tensor;
use coeus_core::{ComputeBackend, Scalar, Shape};

impl<T: Scalar, B: ComputeBackend> Tensor<T, B> {
    /// Zero-copy slice. Returns a view sharing the same storage.
    ///
    /// `ranges` is a slice of `(start, end)` pairs, one per dimension.
    ///
    /// # Panics
    /// If ranges length ≠ ndim, or any range is invalid.
    #[inline]
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Self {
        Self {
            storage: self.storage.clone(),
            layout: self.layout.slice(ranges),
            _backend: PhantomData,
        }
    }

    /// Zero-copy transpose of a 2-D tensor.
    ///
    /// # Panics
    /// If ndim ≠ 2.
    #[inline]
    pub fn t(&self) -> Self {
        assert_eq!(self.ndim(), 2, "transpose requires 2D tensor");
        self.permute(&[1, 0])
    }

    /// Zero-copy transpose of the last two dimensions (batched transpose).
    ///
    /// Swaps `shape[-2]` and `shape[-1]` (and their strides).  For 2-D
    /// tensors this is identical to `t()`.
    ///
    /// # Panics
    /// If ndim < 2.
    #[inline]
    pub fn t_nd(&self) -> Self {
        let nd = self.ndim();
        assert!(nd >= 2, "t_nd requires at least a 2-D tensor, got {nd}-D");
        let mut dims = (0..nd).collect::<Vec<_>>();
        dims.swap(nd - 2, nd - 1);
        self.permute(&dims)
    }

    /// Zero-copy reshape (requires contiguous).
    ///
    /// # Panics
    /// If total elements don't match, or tensor is non-contiguous.
    #[inline]
    pub fn reshape<S: Into<Shape>>(&self, new_shape: S) -> Self {
        let new_shape = new_shape.into();
        let layout = coeus_leto::reshape_layout(&self.layout, &new_shape)
            .expect("coeus-leto reshape validation failed");
        Self {
            storage: self.storage.clone(),
            layout,
            _backend: PhantomData,
        }
    }

    /// Zero-copy permute: re-order dimensions.
    ///
    /// `dims` specifies the new order of dimensions.
    #[inline]
    pub fn permute(&self, dims: &[usize]) -> Self {
        let layout = coeus_leto::permute_layout(&self.layout, dims)
            .expect("coeus-leto permute validation failed");
        Self {
            storage: self.storage.clone(),
            layout,
            _backend: PhantomData,
        }
    }

    /// Zero-copy broadcast to a target shape.
    ///
    /// # Panics
    /// If shapes are not broadcast-compatible.
    #[inline]
    pub fn broadcast<S: Into<Shape>>(&self, target_shape: S) -> Self {
        let target_shape = target_shape.into();
        let layout = coeus_leto::broadcast_layout(&self.layout, &target_shape)
            .expect("coeus-leto broadcast validation failed");

        Self {
            storage: self.storage.clone(),
            layout,
            _backend: PhantomData,
        }
    }

    /// Zero-copy squeeze of a specific dimension of size 1.
    #[inline]
    pub fn squeeze(&self, axis: usize) -> Self {
        Self {
            storage: self.storage.clone(),
            layout: self.layout.squeeze(axis),
            _backend: PhantomData,
        }
    }

    /// Zero-copy squeeze of all dimensions of size 1.
    #[inline]
    pub fn squeeze_all(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            layout: self.layout.squeeze_all(),
            _backend: PhantomData,
        }
    }

    /// Zero-copy unsqueeze by inserting a dimension of size 1 at `axis`.
    #[inline]
    pub fn unsqueeze(&self, axis: usize) -> Self {
        Self {
            storage: self.storage.clone(),
            layout: self.layout.unsqueeze(axis),
            _backend: PhantomData,
        }
    }
}

/// Transpose marker trait (used in matmul autograd).
pub trait Transpose {
    fn transpose(&self) -> Self;
}

impl<T: Scalar, B: ComputeBackend> Transpose for Tensor<T, B> {
    fn transpose(&self) -> Self {
        self.t()
    }
}
