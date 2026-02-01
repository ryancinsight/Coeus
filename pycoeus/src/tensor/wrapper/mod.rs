use backend::CpuBackend;
#[cfg(feature = "gpu")]
use backend::GpuBackend;
use dtype::float::{Float32, Float64};
use dtype::int::Int64;
use storage::{CsrStorage, DenseStorage, StridedStorage};
use tensor::tensor_core::Tensor;

pub mod core_ops;
pub mod linalg;
pub mod manipulation;
pub mod math;

#[derive(Clone, Debug)]
pub enum TensorWrapper {
    CpuDenseF32(Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuDenseF64(Tensor<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuDenseF32(Tensor<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuSparseF32(Tensor<CpuBackend<Float32>, CsrStorage<Float32>, Float32>),
    CpuSparseF64(Tensor<CpuBackend<Float64>, CsrStorage<Float64>, Float64>),
    CpuDenseI64(Tensor<CpuBackend<Int64>, DenseStorage<Int64>, Int64>),
    CpuDenseC32(Tensor<CpuBackend<dtype::complex::Complex32>, DenseStorage<dtype::complex::Complex32>, dtype::complex::Complex32>),
    // Strided variants for views
    CpuStridedF32(Tensor<CpuBackend<Float32>, StridedStorage<Float32>, Float32>),
    CpuStridedF64(Tensor<CpuBackend<Float64>, StridedStorage<Float64>, Float64>),
    CpuStridedI64(Tensor<CpuBackend<Int64>, StridedStorage<Int64>, Int64>),
    #[cfg(feature = "gpu")]
    GpuStridedF32(Tensor<GpuBackend<Float32>, StridedStorage<Float32>, Float32>),
    CpuStridedC32(Tensor<CpuBackend<dtype::complex::Complex32>, StridedStorage<dtype::complex::Complex32>, dtype::complex::Complex32>),
}

pub trait WrapTensor {
    fn wrap(self) -> TensorWrapper;
}

impl WrapTensor for Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::CpuDenseF32(self) }
}
impl WrapTensor for Tensor<CpuBackend<Float64>, DenseStorage<Float64>, Float64> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::CpuDenseF64(self) }
}
#[cfg(feature = "gpu")]
impl WrapTensor for Tensor<GpuBackend<Float32>, DenseStorage<Float32>, Float32> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::GpuDenseF32(self) }
}
impl WrapTensor for Tensor<CpuBackend<Float32>, CsrStorage<Float32>, Float32> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::CpuSparseF32(self) }
}
impl WrapTensor for Tensor<CpuBackend<Float64>, CsrStorage<Float64>, Float64> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::CpuSparseF64(self) }
}
impl WrapTensor for Tensor<CpuBackend<Int64>, DenseStorage<Int64>, Int64> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::CpuDenseI64(self) }
}
impl WrapTensor for Tensor<CpuBackend<dtype::complex::Complex32>, DenseStorage<dtype::complex::Complex32>, dtype::complex::Complex32> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::CpuDenseC32(self) }
}
impl WrapTensor for Tensor<CpuBackend<Float32>, StridedStorage<Float32>, Float32> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::CpuStridedF32(self) }
}
impl WrapTensor for Tensor<CpuBackend<Float64>, StridedStorage<Float64>, Float64> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::CpuStridedF64(self) }
}
impl WrapTensor for Tensor<CpuBackend<Int64>, StridedStorage<Int64>, Int64> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::CpuStridedI64(self) }
}
#[cfg(feature = "gpu")]
impl WrapTensor for Tensor<GpuBackend<Float32>, StridedStorage<Float32>, Float32> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::GpuStridedF32(self) }
}
impl WrapTensor for Tensor<CpuBackend<dtype::complex::Complex32>, StridedStorage<dtype::complex::Complex32>, dtype::complex::Complex32> {
    fn wrap(self) -> TensorWrapper { TensorWrapper::CpuStridedC32(self) }
}
