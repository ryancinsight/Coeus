//! Element-wise mathematical operations.
//!
//! This module provides convenience methods for element-wise mathematical functions
//! that delegate to the single source of truth in `ops::arithmetic`.
//!
//! **ARCHITECTURAL NOTE:** This module follows the Single Source of Truth (SSOT) principle.
//! All implementations are in `tensor::ops::arithmetic`. These methods are thin wrappers
//! that provide ergonomic method-call syntax while delegating to the stateless functions.

use num_traits::Float;
use crate::FloatExt;
use crate::StorageToDense;

/// Element-wise mathematical operations for tensors with float-extended types.
///
/// These methods delegate to `tensor::ops::arithmetic` functions, maintaining
/// Single Source of Truth (SSOT) principle.
impl<B, S, T> crate::Tensor<B, S, T>
where
    B: crate::Backend<Data = T> + Clone + Send + Sync + Default + 'static,
    S: crate::Storage<T> + Clone + crate::StorageFromVec<T> + 'static,
    T: crate::DataType + Float + Clone,
{
    /// Computes the exponential of each element in the tensor.
    ///
    /// Delegates to `tensor::ops::arithmetic::exp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(0.0), Float32::new(1.0)],
    ///     &[2]
    /// ).unwrap();
    ///
    /// let b = a.exp();
    /// // exp(0) = 1, exp(1) ≈ 2.718
    /// ```
    #[must_use]
    pub fn exp(&self) -> Self 
    where
        T: FloatExt,
        S: StorageToDense<T> + crate::ops::TensorStorageOps<T>,
    {
        crate::ops::exp(self)
            .expect("exp operation failed: this is a bug in the tensor implementation")
    }

    /// Computes the natural logarithm of each element in the tensor.
    ///
    /// Delegates to `tensor::ops::arithmetic::log`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(1.0), Float32::new(std::f32::consts::E)],
    ///     &[2]
    /// ).unwrap();
    ///
    /// let b = a.log();
    /// // log(1) = 0, log(e) = 1
    /// ```
    #[must_use]
    pub fn log(&self) -> Self 
    where
        T: FloatExt,
        S: StorageToDense<T> + crate::ops::TensorStorageOps<T>,
    {
        crate::ops::log(self)
            .expect("log operation failed: this is a bug in the tensor implementation")
    }

    /// Computes the sine of each element in the tensor.
    ///
    /// Delegates to `tensor::ops::arithmetic::sin`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(0.0), Float32::new(std::f32::consts::PI / 2.0)],
    ///     &[2]
    /// ).unwrap();
    ///
    /// let b = a.sin();
    /// // sin(0) = 0, sin(π/2) ≈ 1
    /// ```
    #[must_use]
    pub fn sin(&self) -> Self 
    where
        T: FloatExt,
        S: StorageToDense<T> + crate::ops::TensorStorageOps<T>,
    {
        crate::ops::sin(self)
            .expect("sin operation failed: this is a bug in the tensor implementation")
    }

    /// Computes the cosine of each element in the tensor.
    ///
    /// Delegates to `tensor::ops::arithmetic::cos`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_slice(
    ///     &[Float32::new(0.0), Float32::new(std::f32::consts::PI / 2.0)], &[2]
    /// ).unwrap();
    ///
    /// let b = a.cos();
    /// // cos(0) = 1, cos(π/2) ≈ 0
    /// ```
    #[must_use]
    pub fn cos(&self) -> Self 
    where
        T: FloatExt,
        S: StorageToDense<T> + crate::ops::TensorStorageOps<T>,
    {
        crate::ops::cos(self)
            .expect("cos operation failed: this is a bug in the tensor implementation")
    }

    /// Computes the power of each element in the tensor.
    ///
    /// Delegates to `tensor::ops::arithmetic::pow_scalar`.
    ///
    /// # Arguments
    ///
    /// * `exp` - The exponent to raise each element to
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(2.0), Float32::new(3.0)],
    ///     &[2]
    /// ).unwrap();
    ///
    /// let b = a.powf(Float32::new(2.0));
    /// // 2^2 = 4, 3^2 = 9
    /// ```
    #[must_use]
    pub fn powf(&self, exp: T) -> Self
    where
        T: num_traits::Num + FloatExt,
        S: StorageToDense<T> + crate::ops::TensorStorageOps<T>,
    {
        crate::ops::math::pow_scalar(self, exp)
            .expect("powf operation failed: this is a bug in the tensor implementation")
    }

    /// Computes the square of each element in the tensor.
    ///
    /// This is a convenience method that uses `powf(2.0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(2.0), Float32::new(3.0)],
    ///     &[2]
    /// ).unwrap();
    ///
    /// let b = a.square();
    /// // 2² = 4, 3² = 9
    /// ```
    #[must_use]
    pub fn square(&self) -> Self
    where
        T: num_traits::Num + num_traits::FromPrimitive + FloatExt,
        S: StorageToDense<T> + crate::ops::TensorStorageOps<T>,
    {
        let two = T::from_f64(2.0).expect("Failed to convert 2.0 to target type");
        crate::ops::math::pow_scalar(self, two)
            .expect("square operation failed: this is a bug in the tensor implementation")
    }

    /// Computes the square root of each element in the tensor.
    ///
    /// Delegates to `tensor::ops::arithmetic::sqrt`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(4.0), Float32::new(9.0)],
    ///     &[2]
    /// ).unwrap();
    ///
    /// let b = a.sqrt();
    /// // √4 = 2, √9 = 3
    /// ```
    #[must_use]
    pub fn sqrt(&self) -> Self 
    where
        T: FloatExt + num_traits::FromPrimitive,
        S: StorageToDense<T> + crate::ops::TensorStorageOps<T>,
    {
        crate::ops::sqrt(self)
            .expect("sqrt operation failed: this is a bug in the tensor implementation")
    }
}
