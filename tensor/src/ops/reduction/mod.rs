//! Reduction operations module

mod all;
mod any;
mod max;
mod mean;
mod min;
mod std;
mod sum;
mod var;

pub use all::all;
pub use any::any;
pub use max::max;
pub use mean::mean;
pub use min::min;
pub use self::std::std;
pub use sum::sum;
pub use var::var;

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
