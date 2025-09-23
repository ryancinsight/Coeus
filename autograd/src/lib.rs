//! # Coeus Autograd
//!
//! Automatic differentiation engine providing computational graphs, gradient computation,
//! and higher-order derivatives (Hessian matrices) for second-order optimization.
//!
//! This crate implements reverse-mode automatic differentiation with support for:
//! - Computational graphs with topological sorting
//! - Gradient accumulation and backpropagation
//! - Higher-order derivatives using finite differences
//! - Thread-safe gradient computation
//! - Memory-efficient graph construction
//! - Hessian matrix computation and iteration

pub mod context;
pub mod differentiable;
pub mod edge_case_tests;
pub mod graph;
pub mod numerical_stability;
pub mod ops;
pub mod tensor_ref;

// Re-exports for public API
pub use context::{AutogradContext, Operation as ContextOperation};
pub use differentiable::{AutogradError, Context, Differentiable, GradientAccumulator};
pub use graph::{ComputationalGraph, HessianIter, Node, NodeId, Operation};
pub use ops::BackwardOp;
pub use tensor_ref::TensorRef;
