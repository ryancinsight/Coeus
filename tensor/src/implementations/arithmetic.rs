use crate::{Backend, DataType, Storage, Tensor};
use storage::{StorageFromVec, StorageToDense};

// Add
impl<B, S, T> std::ops::Add<Tensor<B, S, T>> for Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Add<Output = T> + Copy + 'static,
{
    type Output = Tensor<B, S, T>;

    fn add(self, rhs: Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::add(&self, &rhs).unwrap()
    }
}

impl<'a, B, S, T> std::ops::Add<&'a Tensor<B, S, T>> for &'a Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Add<Output = T> + Copy + 'static,
{
    type Output = Tensor<B, S, T>;

    fn add(self, rhs: &'a Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::add(self, rhs).unwrap()
    }
}

// Sub
impl<B, S, T> std::ops::Sub<Tensor<B, S, T>> for Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Sub<Output = T> + std::ops::Neg<Output = T> + Copy + 'static,
{
    type Output = Tensor<B, S, T>;

    fn sub(self, rhs: Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::sub(&self, &rhs).unwrap()
    }
}

impl<'a, B, S, T> std::ops::Sub<&'a Tensor<B, S, T>> for &'a Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Sub<Output = T> + std::ops::Neg<Output = T> + Copy + 'static,
{
    type Output = Tensor<B, S, T>;

    fn sub(self, rhs: &'a Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::sub(self, rhs).unwrap()
    }
}

// Mul
impl<B, S, T> std::ops::Mul<Tensor<B, S, T>> for Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + std::ops::Neg<Output = T> + Copy + Default + num_traits::Zero + 'static,
{
    type Output = Tensor<B, S, T>;

    fn mul(self, rhs: Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::mul(&self, &rhs).unwrap()
    }
}

impl<'a, B, S, T> std::ops::Mul<&'a Tensor<B, S, T>> for &'a Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + std::ops::Neg<Output = T> + Copy + Default + num_traits::Zero + 'static,
{
    type Output = Tensor<B, S, T>;

    fn mul(self, rhs: &'a Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::mul(self, rhs).unwrap()
    }
}

// Div
impl<B, S, T> std::ops::Div<Tensor<B, S, T>> for Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Div<Output = T> + Copy + 'static + std::ops::Neg<Output = T>,
{
    type Output = Tensor<B, S, T>;

    fn div(self, rhs: Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::div(&self, &rhs).unwrap()
    }
}

impl<'a, B, S, T> std::ops::Div<&'a Tensor<B, S, T>> for &'a Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Div<Output = T> + Copy + 'static + std::ops::Neg<Output = T>,
{
    type Output = Tensor<B, S, T>;

    fn div(self, rhs: &'a Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::div(self, rhs).unwrap()
    }
}

// Neg
impl<B, S, T> std::ops::Neg for Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Neg<Output = T> + Copy + 'static,
{
    type Output = Tensor<B, S, T>;

    fn neg(self) -> Self::Output {
        crate::ops::arithmetic::neg(&self).unwrap()
    }
}

impl<'a, B, S, T> std::ops::Neg for &'a Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Neg<Output = T> + Copy + 'static,
{
    type Output = Tensor<B, S, T>;

    fn neg(self) -> Self::Output {
        crate::ops::arithmetic::neg(self).unwrap()
    }
}
