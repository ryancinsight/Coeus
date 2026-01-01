use backend::Backend;
use dtype::DataType;
use num_traits::{Float, Zero};
use storage::{Storage, StorageFromVec};
use tensor::Tensor;

pub trait Norm<B, S, T> {
    fn norm(&self) -> coeus_error::Result<T>; // Default Frobenius
    fn norm_p(&self, p: T) -> coeus_error::Result<T>;
}

impl<B, S, T> Norm<B, S, T> for Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + Float + Zero + std::iter::Sum + std::ops::AddAssign,
{
    fn norm(&self) -> coeus_error::Result<T> {
        // Frobenius norm: sqrt(sum(x^2))
        let iter = self.as_slice().iter();
        let sum_sq: T = iter.map(|&x| x * x).sum();
        Ok(sum_sq.sqrt())
    }

    fn norm_p(&self, p: T) -> coeus_error::Result<T> {
        let iter = self.as_slice().iter();
        let sum_pow: T = iter.map(|&x| x.powf(p)).sum();
        Ok(sum_pow.powf(T::one() / p))
    }
}
