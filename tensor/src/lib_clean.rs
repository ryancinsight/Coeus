//! # Coeus Tensor
//!
//! Core tensor implementation with nested type hierarchy: `Tensor<B<S<T>>>`
//!
//! ## Architecture
//!
//! Tensors compose three orthogonal abstractions via nested generics:
//!
//! ```text
//! Tensor<B, S, T>
//! ├── B: Backend      // Compute device (CPU, GPU, NPU)
//! ├── S: Storage<T>   // Memory layout (Dense, Sparse, Strided)
//! └── T: DataType     // Element type (f32, i32, Complex, etc.)
//! ```
//!
//! ## Automatic Differentiation
//!
//! Tensors support PyTorch-compatible automatic differentiation:
//!
//! ```text
//! Tensor<B, S, T>
//! ├── requires_grad: bool     // Enable gradient computation
//! ├── grad: Option<Tensor>    // Accumulated gradients
//! └── grad_fn: Option<Function> // Function that created this tensor
//! ```
//!
//! ## Design (ADR-001)
//!
//! Nested trait bounds enable:
//! - **Zero-cost dispatch**: Monomorphization eliminates runtime overhead
//! - **Type safety**: Compile-time guarantees prevent dtype mismatches
//! - **Extensibility**: New backends/storage/types via trait impls

// Core infrastructure
pub mod error;
pub mod tensor_impl;
pub mod tensor_sparse;
pub mod tensor_core;
pub mod tensor_autograd;
pub mod tensor_dense_ops;
pub mod tensor_dense_ops_ext;
pub mod tensor_sparse_ops;

// Tensor operations
pub mod creation;
pub mod indexing;
pub mod shape_ops;

// Mathematical operations
pub mod arithmetic;
pub mod elementwise;
pub mod matrix;
pub mod reduction;

// Re-export core types and traits
pub use tensor_core::{AsAny, DifferentiableFunction, Function, Backend, CpuBackend};
pub use tensor_core::Tensor;
pub use tensor_core::Device;

// Re-export storage and dtype traits
pub use coeus_dtype::traits::FloatExt;
pub use coeus_dtype::DataType;
pub use coeus_storage::{DenseStorage, Shape, Storage, StorageFromVec, StorageToDense};

// Error handling
pub use error::TensorError;

// Result type for tensor operations
pub type Result<T> = core::result::Result<T, TensorError>;

// PyTorch-style type aliases for ergonomic API
/// CPU dense tensor type alias
pub type TensorCpuDense<T> = Tensor<CpuBackend<T>, DenseStorage<T>, T>;

// Re-export AutoGradTensor from the autograd module
pub use tensor_autograd::AutoGradTensor;

