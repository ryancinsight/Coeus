#![deny(missing_docs)]

//! Distributed training primitives for the Coeus framework.
//!
//! Provides a [`Communicator`] trait abstracting process-group communication, with
//! a thread-based [`LocalCommunicator`] for single-process verification and a socket-based
//! [`TcpCommunicator`]/[`TcpMesh`] for real multi-process runs. Gradient averaging is
//! exposed via [`synchronize_gradients`].

/// Collective communication interface implemented by all communicators.
pub mod communicator;
/// Data-parallel gradient averaging over a [`Communicator`].
pub mod gradients;
pub(crate) mod host_access;
/// Thread-based simulated communicator for local multi-process verification.
pub mod local;
/// Zero-sized reduction-operation tags (`Sum`, `Min`, `Max`, `Product`).
pub mod ops;
/// Socket-based communicators for real multi-process training.
pub mod tcp;

pub use communicator::Communicator;
pub use gradients::{synchronize_gradients, GradientSyncError};
pub use local::LocalCommunicator;
pub use ops::{Max, Min, Product, ReduceOpTag, Sum};
pub use tcp::{MeshDeadlines, StreamStep, TcpCommunicator, TcpMesh, TcpMeshError};
