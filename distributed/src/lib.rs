//! # Distributed Training for Coeus
//!
//! This crate provides distributed training capabilities for multi-GPU and multi-node
//! training scenarios. It implements data parallelism with gradient synchronization
//! across devices.
//!
//! ## Architecture
//!
//! The distributed training system is built around:
//! - **Data Parallelism**: Model replication across devices with data batch splitting
//! - **Gradient Synchronization**: AllReduce operations for gradient aggregation
//! - **Process Groups**: Communication collectives for inter-device coordination
//! - **Distributed Optimizers**: Gradient synchronization wrappers for existing optimizers

pub mod communication;
pub mod data_parallel;
pub mod error;
pub mod optimizer;
pub mod process_group;
pub mod reducer;

pub use communication::{CommunicationBackend, BackendType, BackendStats, NCCLBackend, GlooBackend, MPIBackend, create_backend, create_auto_backend};
pub use data_parallel::DataParallel;
pub use error::{DistributedError, Result};
pub use optimizer::DistributedOptimizer;
pub use process_group::{ProcessGroup, Rank, WorldSize, FaultToleranceConfig};
pub use reducer::GradientReducer;
