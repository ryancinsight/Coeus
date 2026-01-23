//! Reduction implementations for Tensor

use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Base reduction implementation along specified dimensions.
    pub fn reduce_generic<F>(
        &self,
        dims: Option<&[usize]>,
        keepdim: bool,
        op: F,
        init: T,
    ) -> Result<Tensor<B, S, T>>
    where
        F: Fn(T, T) -> T,
    {
        let shape = self.shape().dims();
        let mut reduce_dims = dims.map(|d| d.to_vec()).unwrap_or_else(|| (0..shape.len()).collect());
        reduce_dims.sort_unstable();
        reduce_dims.dedup();

        if reduce_dims.iter().any(|&d| d >= shape.len()) {
            return Err(TensorError::InvalidDimension {
                dim: *reduce_dims.iter().find(|&&d| d >= shape.len()).unwrap(),
                ndim: shape.len(),
            });
        }

        let mut out_shape = Vec::new();
        for (i, &d) in shape.iter().enumerate() {
            if reduce_dims.contains(&i) {
                if keepdim {
                    out_shape.push(1);
                }
            } else {
                out_shape.push(d);
            }
        }

        let out_numel: usize = out_shape.iter().product();
        let mut out_data = vec![init; out_numel];

        let stride: Vec<usize> = (0..shape.len())
            .map(|i| shape.iter().skip(i + 1).product())
            .collect();
        let out_stride: Vec<usize> = (0..out_shape.len())
            .map(|i| out_shape.iter().skip(i + 1).product())
            .collect();

        let data = self.as_slice();
        for i in 0..data.len() {
            let mut out_idx = 0;
            let mut temp_idx = i;
            let mut out_dim_idx = 0;

            for (d, &s) in stride.iter().enumerate() {
                let coord = temp_idx / s;
                temp_idx %= s;

                if !reduce_dims.contains(&d) {
                    out_idx += coord * out_stride[out_dim_idx];
                    out_dim_idx += 1;
                } else if keepdim {
                    // skip out_idx update as coord is effectively 0 for a reduced dim of size 1
                    out_dim_idx += 1;
                }
            }
            out_data[out_idx] = op(out_data[out_idx], data[i]);
        }

        Tensor::from_vec_with_backend(out_data, &out_shape, self.backend.clone())
    }

    /// Sum reduction along specified dimensions.
    pub fn sum_generic(&self, dims: Option<&[usize]>, keepdim: bool) -> Result<Tensor<B, S, T>>
    where
        T: core::ops::Add<Output = T>,
    {
        self.reduce_generic(dims, keepdim, |acc, x| acc + x, T::zero())
    }
}
