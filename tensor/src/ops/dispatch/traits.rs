//! Unified storage operations trait.
//!
//! `TensorStorageOps` is the single trait that all storage types implement
//! to enable uniform tensor operations. It combines:
//! - Arithmetic operations (add, sub, mul, div, neg)
//! - Linear algebra operations (matmul, transpose)
//! - Transcendental operations (exp, log, sin, cos, etc.)
//! - Activation functions (relu)
//! - Reduction operations (sum, mean, max, min)
//!
//! ## Design Philosophy
//!
//! Following the B,S,T pattern (Backend, Storage, Type), this trait:
//! - Accepts `backend: &B` for hardware dispatch
//! - Returns `Result<Self>` to preserve storage type
//! - Uses `num_traits` bounds for mathematical operations

use crate::Result;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, StorageFormat};

/// Unified trait for storage-level operations.
///
/// This trait provides a single interface for all tensor operations,
/// enabling zero-cost dispatch based on storage type and backend.
pub trait TensorStorageOps<T: DataType>: storage::Storage<T> + storage::StorageToDense<T> + Sized {
    // ================== Arithmetic Operations ==================

    /// Element-wise addition: self + other
    fn storage_add<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>;

    /// Element-wise subtraction: self - other
    fn storage_sub<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>;

    /// Element-wise multiplication: self * other
    fn storage_mul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>;

    /// Element-wise division: self / other
    fn storage_div<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>;

    /// Element-wise negation: -self
    fn storage_neg<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: core::ops::Neg<Output = T>;

    // ================== Comparison Operations ==================

    /// Element-wise equality: self == other
    fn storage_eq<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero;

    /// Element-wise inequality: self != other
    fn storage_ne<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero;

    /// Element-wise greater than: self > other
    fn storage_gt<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero;

    /// Element-wise greater or equal: self >= other
    fn storage_ge<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero;

    /// Element-wise less than: self < other
    fn storage_lt<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero;

    /// Element-wise less or equal: self <= other
    fn storage_le<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero;

    // ================== Linear Algebra Operations ==================

    /// Matrix-matrix multiplication: self @ other
    fn storage_matmul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>;

    /// Matrix transpose
    fn storage_transpose<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>;

    /// Add matrix multiplication: beta * self + alpha * (mat1 @ mat2)
    fn storage_addmm<B: Backend<Data = T>>(
        &self,
        _mat1: &Self,
        _mat2: &Self,
        _beta: T,
        _alpha: T,
        _backend: &B
    ) -> Result<Self> {
         Err(crate::TensorError::UnsupportedOperation {
             operation: "storage_addmm".to_string(),
             storage_type: "Generic".to_string(),
         })
    }

    /// Add matrix-vector multiplication: beta * self + alpha * (mat @ vec)
    fn storage_addmv<B: Backend<Data = T>>(
        &self,
        _mat: &Self,
        _vec: &Self,
        _beta: T,
        _alpha: T,
        _backend: &B
    ) -> Result<Self> {
         Err(crate::TensorError::UnsupportedOperation {
             operation: "storage_addmv".to_string(),
             storage_type: "Generic".to_string(),
         })
    }

    // ================== Activation Functions ==================

    /// ReLU activation: max(0, x)
    fn storage_relu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + Default;

    /// Sigmoid activation: 1 / (1 + exp(-x))
    fn storage_sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    fn storage_tanh<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// GELU activation
    fn storage_gelu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    // ================== Reduction Operations ==================

    /// Sum of all elements
    fn storage_sum<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>;

    /// Mean of all elements
    fn storage_mean<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: num_traits::FromPrimitive;

    /// Maximum element
    fn storage_max<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: PartialOrd;

    /// Minimum element
    fn storage_min<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: PartialOrd;

    // ================== Transcendental Operations ==================

    /// Element-wise exponential
    fn storage_exp<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise natural logarithm
    fn storage_log<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise sine
    fn storage_sin<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise cosine
    fn storage_cos<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise absolute value
    fn storage_abs<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Signed;

    /// Element-wise ceiling
    fn storage_ceil<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise floor
    fn storage_floor<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise rounding
    fn storage_round<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise square root
    fn storage_sqrt<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise reciprocal square root
    fn storage_rsqrt<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise error function
    fn storage_erf<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise complementary error function
    fn storage_erfc<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise inverse error function
    fn storage_erfinv<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise inverse tangent (y, x)
    fn storage_atan2<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise log1p: log(1 + x)
    fn storage_log1p<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise expm1: exp(x) - 1
    fn storage_expm1<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    /// Element-wise reciprocal: 1 / x
    fn storage_reciprocal<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float;

    // ================== Comparison/Status Operations ==================

    /// Element-wise check for NaN
    fn storage_isnan<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero;

    /// Element-wise check for infinity
    fn storage_isinf<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero;

    /// Element-wise check for finite values
    fn storage_isfinite<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero;

    // ================== Logical Operations ==================

    /// Element-wise logical AND
    fn storage_logical_and<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero;

    /// Element-wise logical OR
    fn storage_logical_or<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero;

    /// Element-wise logical XOR
    fn storage_logical_xor<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero;

    /// Element-wise logical NOT
    fn storage_logical_not<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero;

    // ================== Conversion Operations ==================

    /// Convert to dense storage
    fn storage_to_dense(&self) -> Result<DenseStorage<T>>
    where
        T: num_traits::Zero + Clone;

    /// Get storage format for runtime dispatch
    fn storage_format(&self) -> StorageFormat {
        self.format()
    }
}

/// Trait for mixed-storage binary operations.
///
/// This trait enables zero-conversion operations between different storage types
/// (e.g., Dense + Sparse) by providing a double-dispatch mechanism.
pub trait StorageBinaryOps<OtherS, T: DataType>: storage::Storage<T> + Sized {
    /// The resulting storage type of the mixed operation
    type Output: storage::Storage<T> + Sized;

    /// Mixed-storage element-wise addition: self + other
    fn storage_add_mixed<B: Backend<Data = T>>(&self, other: &OtherS, backend: &B) -> Result<Self::Output>;

    /// Mixed-storage element-wise subtraction: self - other
    fn storage_sub_mixed<B: Backend<Data = T>>(&self, other: &OtherS, backend: &B) -> Result<Self::Output>;

    /// Mixed-storage element-wise multiplication: self * other
    fn storage_mul_mixed<B: Backend<Data = T>>(&self, other: &OtherS, backend: &B) -> Result<Self::Output>;

    /// Mixed-storage element-wise division: self / other
    fn storage_div_mixed<B: Backend<Data = T>>(&self, other: &OtherS, backend: &B) -> Result<Self::Output>;

    /// Mixed-storage matrix multiplication: self @ other
    fn storage_matmul_mixed<B: Backend<Data = T>>(&self, other: &OtherS, backend: &B) -> Result<Self::Output>;
}
