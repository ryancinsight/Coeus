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
//! - **grad_fn Chain**: PyTorch-compatible `tensor.grad_fn.next_functions` navigation
//! - **Lazy Gradient Computation**: Gradients computed only when `.backward()` is called
//!
//! ## Memory Efficiency
//!
//! Unlike the abandoned operation-based system (100MB+ per Conv2D), the Function-based
//! approach uses O(1) memory per operation with lightweight tensor references.
//!
//! ## Example
//!
//! ```rust
//! use coeus_tensor::{TensorCpuDense, AutoGradTensor};
//! use coeus_dtype::float::Float32;
//!
//! // Create tensors with gradient tracking
//! let x = TensorCpuDense::<Float32>::from_vec(vec![
//!     Float32::new(2.0), Float32::new(3.0)
//! ], &[2]).unwrap().requires_grad_(true);
//! let x_autograd = AutoGradTensor::new(x);
//!
//! // Automatic graph construction during operations
//! let y = x_autograd.exp().sum().unwrap();
//!
//! // Automatic gradient computation
//! y.backward().unwrap();
//!
//! // Access gradients via PyTorch-compatible API
//! println!("x.grad: {:?}", x_autograd.grad()); // [e², e³]
//! ```

#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod computation_graph;
pub mod custom;
pub mod functions;
pub mod graph_node;
pub mod loss;
pub mod nn;
pub mod numerical;
pub mod tensor_ops;
pub mod ops;
pub mod checkpointing;

// Re-export key types for ergonomics
pub use computation_graph::GradientEngine;
pub use coeus_tensor::{Function, DifferentiableFunction};
pub use custom::apply_custom_function;

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
pub use ops::{grad, hvp, jvp, backward, backward_with_grad, backward_with_grad_and_options};

// Re-export graph construction types
pub use graph_node::{GraphNode, NodeId, NodeRegistry, TopologicalSorter, GradientAccumulator};

// Re-export function types
pub use functions::FunctionRef;

// Re-export gradient checkpointing
pub use checkpointing::{checkpoint, checkpoint_sequential};
