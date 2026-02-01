use crate::{Backend, DataType, Storage, Tensor};
use storage::{StorageFromVec, StorageToDense};

// Add
impl<B, S1, S2, T> std::ops::Add<Tensor<B, S2, T>> for Tensor<B, S1, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S1, T> + 'static,
    S1: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    S2: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Add<Output = T> + Copy + 'static,
{
    type Output = Tensor<B, S1, T>;

    fn add(self, rhs: Tensor<B, S2, T>) -> Self::Output {
        crate::ops::arithmetic::add(&self, &rhs).unwrap()
    }
}

impl<'a, B, S1, S2, T> std::ops::Add<&'a Tensor<B, S2, T>> for &'a Tensor<B, S1, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S1, T> + 'static,
    S1: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    S2: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Add<Output = T> + Copy + 'static,
{
    type Output = Tensor<B, S1, T>;

    fn add(self, rhs: &'a Tensor<B, S2, T>) -> Self::Output {
        crate::ops::arithmetic::add(self, rhs).unwrap()
    }
}

// Sub
impl<B, S1, S2, T> std::ops::Sub<Tensor<B, S2, T>> for Tensor<B, S1, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S1, T> + 'static,
    S1: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    S2: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Sub<Output = T> + std::ops::Neg<Output = T> + Copy + 'static,
{
    type Output = Tensor<B, S1, T>;

    fn sub(self, rhs: Tensor<B, S2, T>) -> Self::Output {
        crate::ops::arithmetic::sub(&self, &rhs).unwrap()
    }
}

impl<'a, B, S1, S2, T> std::ops::Sub<&'a Tensor<B, S2, T>> for &'a Tensor<B, S1, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S1, T> + 'static,
    S1: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    S2: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Sub<Output = T> + std::ops::Neg<Output = T> + Copy + 'static,
{
    type Output = Tensor<B, S1, T>;

    fn sub(self, rhs: &'a Tensor<B, S2, T>) -> Self::Output {
        crate::ops::arithmetic::sub(self, rhs).unwrap()
    }
}

// Mul
impl<B, S1, S2, T> std::ops::Mul<Tensor<B, S2, T>> for Tensor<B, S1, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S1, T> + 'static,
    S1: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    S2: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + std::ops::Neg<Output = T> + Copy + Default + num_traits::Zero + 'static,
{
    type Output = Tensor<B, S1, T>;

    fn mul(self, rhs: Tensor<B, S2, T>) -> Self::Output {
        crate::ops::arithmetic::mul(&self, &rhs).unwrap()
    }
}

impl<'a, B, S1, S2, T> std::ops::Mul<&'a Tensor<B, S2, T>> for &'a Tensor<B, S1, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S1, T> + 'static,
    S1: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    S2: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + std::ops::Neg<Output = T> + Copy + Default + num_traits::Zero + 'static,
{
    type Output = Tensor<B, S1, T>;

    fn mul(self, rhs: &'a Tensor<B, S2, T>) -> Self::Output {
        crate::ops::arithmetic::mul(self, rhs).unwrap()
    }
}

// Div
impl<B, S1, S2, T> std::ops::Div<Tensor<B, S2, T>> for Tensor<B, S1, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S1, T> + 'static,
    S1: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    S2: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Div<Output = T> + Copy + 'static + std::ops::Neg<Output = T>,
{
    type Output = Tensor<B, S1, T>;

    fn div(self, rhs: Tensor<B, S2, T>) -> Self::Output {
        crate::ops::arithmetic::div(&self, &rhs).unwrap()
    }
}

impl<'a, B, S1, S2, T> std::ops::Div<&'a Tensor<B, S2, T>> for &'a Tensor<B, S1, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + crate::tensor_backend_dispatch::TensorBackendDispatcher<B, S1, T> + 'static,
    S1: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    S2: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + std::ops::Div<Output = T> + Copy + 'static + std::ops::Neg<Output = T>,
{
    type Output = Tensor<B, S1, T>;

    fn div(self, rhs: &'a Tensor<B, S2, T>) -> Self::Output {
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

// Scalar Arithmetic
impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + Copy + Default + 'static + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
{
    pub fn add_scalar(&self, scalar: T) -> crate::Result<Self> {
        crate::ops::arithmetic::add_scalar(self, scalar)
    }

    pub fn sub_scalar(&self, scalar: T) -> crate::Result<Self> {
        crate::ops::arithmetic::sub_scalar(self, scalar)
    }

    pub fn mul_scalar(&self, scalar: T) -> crate::Result<Self> {
        crate::ops::arithmetic::mul_scalar(self, scalar)
    }

    pub fn div_scalar(&self, scalar: T) -> crate::Result<Self> {
        crate::ops::arithmetic::div_scalar(self, scalar)
    }

    pub fn add_scalar_(&mut self, scalar: T) -> crate::Result<()> {
        crate::ops::arithmetic::add_scalar_(&mut self.storage, scalar)
    }

    pub fn sub_scalar_(&mut self, scalar: T) -> crate::Result<()> {
        crate::ops::arithmetic::sub_scalar_(&mut self.storage, scalar)
    }

    pub fn mul_scalar_(&mut self, scalar: T) -> crate::Result<()> {
        crate::ops::arithmetic::mul_scalar_(&mut self.storage, scalar)
    }

    pub fn div_scalar_(&mut self, scalar: T) -> crate::Result<()> {
        crate::ops::arithmetic::div_scalar_(&mut self.storage, scalar)
    }
}
