//! Unified Storage Dispatch
//!
//! This module provides the unified `TensorStorageOps` trait, combining arithmetic,
//! linear algebra, and transcendental operations across all storage types
//! (dense, sparse, quantized) with zero-cost dispatch.
//!
//! ## Architecture
//!
//! The dispatch follows a hierarchical accessor pattern:
//!
//! ```text
//! TensorStorageOps<T>     (This module - unified interface)
//!   ├── Storage.format()  (Runtime format identification)
//!   ├── Arithmetic Ops    (add, sub, mul, div)
//!   ├── LinearAlg Ops     (matmul, matvec)
//!   └── Transcendental    (exp, log, sin, cos, etc.)
//!
//! Dispatch Hierarchy:
//!   Tensor<B, S, T>
//!     └── calls S::storage_add(&self, other, backend)
//!           ├── DenseStorage → dense::DenseArithmetic::add
//!           │     └── Backend::add_dense() → CPU/GPU
//!           └── CsrStorage → sparse::SparseAdd::add_sparse
//!                 └── Pure CPU (future: GPU sparse kernels)
//! ```
//!
//! ## Shared Accessor Pattern
//!
//! All storage types share a common accessor interface, enabling:
//! - Uniform tensor operations regardless of underlying storage
//! - Zero-cost dispatch via trait bounds (no dynamic dispatch)
//! - Future extension to new storage types (e.g., BlockSparse)
//!
//! ## StorageFormat Integration
//!
//! Uses `StorageFormat` enum from storage crate for runtime type identification:
//! - `StorageFormat::Dense` → Dense arithmetic path
//! - `StorageFormat::Csr/Csc/Coo` → Sparse arithmetic path
//! - `StorageFormat::Quantized` → Dequantize-operate-quantize path

pub mod traits;
pub mod dense_dispatch;
pub mod sparse_dispatch;

pub use traits::TensorStorageOps;
