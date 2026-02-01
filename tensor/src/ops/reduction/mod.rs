//! Reduction operations module

mod all;
mod any;
mod argminmax;
mod max;
mod mean;
mod min;
mod std;
mod sum;
mod var;
mod topk;

pub use self::std::std;
pub use all::all;
pub use any::any;
pub use argminmax::{argmax, argmin};
pub use max::max;
pub use mean::mean;
pub use min::min;
pub use sum::sum;
pub use var::var;
pub use topk::topk;

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Broadcasting reduction helper
pub fn reduce_dims<
    T: DataType,
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
    F: Fn(T, T) -> T,
>(
    tensor: &Tensor<B, S, T>,
    dims: Option<&[usize]>,
    keepdim: bool,
    op: F,
    init: T,
) -> Result<Tensor<B, S, T>> {
    tensor.reduce_generic(dims, keepdim, op, init)
}
