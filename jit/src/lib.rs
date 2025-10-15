//! # JIT Compilation & Graph Optimization for Coeus
//!
//! This crate provides just-in-time compilation and graph optimization capabilities
//! for high-performance neural network execution. It implements:
//!
//! - **Computation Graph Construction**: Building efficient graphs from autograd operations
//! - **Graph Optimization**: Dead code elimination, constant folding, operator fusion
//! - **Kernel Fusion**: Combining adjacent operations for reduced memory access
//! - **JIT Compilation**: Runtime code generation for optimized execution
//!
//! ## Architecture
//!
//! The JIT system operates in phases:
//! 1. **Graph Construction**: Convert autograd operations to computation graphs
//! 2. **Optimization**: Apply transformation passes to improve efficiency
//! 3. **Fusion**: Detect and merge fusable operations
//! 4. **Compilation**: Generate optimized machine code
//! 5. **Execution**: Run compiled kernels with caching
//! 6. **Advanced Features**: TorchScript compatibility, dynamic shapes, memory optimization

pub mod cache;
pub mod compiler;
pub mod error;
pub mod fusion;
pub mod graph;
pub mod memory;
pub mod optimizer;
pub mod shapes;
pub mod torchscript;
pub mod tracing;

pub use cache::KernelCache;
pub use compiler::JitCompiler;
pub use error::{JitError, Result};
pub use fusion::{FusedKernel, FusionDetector};
pub use graph::{ComputationGraph, Node, NodeId, Operation};
pub use memory::MemoryArena;
pub use optimizer::{OptimizationPass, Optimizer};
pub use shapes::ShapeSpecializer;
pub use torchscript::TorchScript;
pub use tracing::{TracingContext, TracingGuard};
