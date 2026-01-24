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
pub trait TensorStorageOps<T: DataType>: storage::Storage<T> + Sized {
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
    fn storage_neg<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>;

    // ================== Linear Algebra Operations ==================

    /// Matrix-matrix multiplication: self @ other
    fn storage_matmul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>;

    /// Matrix transpose
    fn storage_transpose<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>;

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
