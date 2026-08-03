//! Feature-aware integration harness for CUDA provider tests.
//!
//! Leaf modules retain their original `cuda`/`not(feature = "cuda")` gates;
//! one Cargo target replaces the previous flat target-per-file topology.

#[path = "cuda_ops/device.rs"]
mod device;
#[cfg(feature = "cuda")]
#[path = "cuda/random_init.rs"]
mod random_init;
#[path = "cuda_ops/unavailable.rs"]
mod unavailable;
