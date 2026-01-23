//! # Coeus Automatic Differentiation
//!
//! This crate implements reverse-mode automatic differentiation for Coeus tensors
//! using PyTorch-compatible dynamic graph construction with automatic differentiation.
//!
//! ## Architecture
//!
//! The autograd system provides PyTorch-compatible automatic differentiation:
//! - **Automatic Graph Construction**: Graphs are built implicitly during tensor operations
//! - **Function Objects**: Lightweight Function trait replaces operation-based approach
//! - **`grad_fn` Chain**: PyTorch-compatible `tensor.grad_fn.next_functions` navigation
//! - **Lazy Gradient Computation**: Gradients computed only when `.backward()` is called
//!
//! ## Memory Efficiency
//!
//! Unlike the abandoned operation-based system (100MB+ per `Conv2D`), the Function-based
//! approach uses O(1) memory per operation with lightweight tensor references.
//!
//! ## Example
//!
//! ```rust
//! use tensor::Tensor;
//! use backend::CpuBackend;
//! use storage::DenseStorage;
//! use dtype::float::Float32;
//!
//! // Create tensors with gradient tracking
//! let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(vec![
//!     Float32::new(2.0), Float32::new(3.0)
//! ], &[2]).unwrap().requires_grad_(true);
//!
//! // Automatic graph construction during operations
//! let y = &x + &x; // Simple operation for demonstration
//!
//! // Gradients are computed automatically when operations are performed
//! println!("Tensor requires gradients: {}", x.requires_grad());
//! ```

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

pub mod checkpointing;
pub mod computation_graph;
pub mod custom;

pub mod functions;
pub mod graph_node;
pub mod loss;
pub mod nn;
pub mod numerical;
/// Autograd operations (backward, grad, hvp, jvp)
pub mod ops;
pub mod sparse_gradients;
pub mod tensor_ops;

#[cfg(test)]
mod tests;

// Re-export key types for ergonomics
pub use custom::apply_custom_function;
pub use storage::AsAny;
pub use tensor::{DifferentiableFunction, Function};

// Core traits for extensibility
pub mod traits {
    //! Traits for extending the autograd system

    use std::any::Any;

    /// Trait for operations that can participate in automatic differentiation
    pub trait Differentiable: Send + Sync {
        /// Compute gradients for this operation's inputs given output gradients
        fn backward(&self, grad_output: &dyn Any) -> Vec<Box<dyn Any>>;
    }

    /// Trait for gradient accumulation strategies
    pub trait GradientAccumulator: Send + Sync {
        /// Accumulate a gradient into the target
        fn accumulate(&mut self, grad: &dyn Any);
    }

    /// Extension trait for downcasting Function objects
    pub trait AsAny {
        /// Get as Any reference for downcasting
        fn as_any(&self) -> &dyn core::any::Any;
    }
}

// Error types
pub mod error;
pub use error::AutogradError;

// Re-export for convenience
pub use error::Result;

// Re-export higher-order derivative functions
pub use ops::{backward, backward_with_grad, backward_with_grad_and_options, grad, hvp, jvp};

// Re-export graph construction types
pub use graph_node::{GradientAccumulator, GraphNode, NodeId, NodeRegistry, TopologicalSorter};

// Re-export function types
pub use functions::FunctionRef;

// Re-export gradient checkpointing
pub use checkpointing::{checkpoint, checkpoint_sequential};
