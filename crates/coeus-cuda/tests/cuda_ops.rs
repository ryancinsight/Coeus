//! Feature-aware integration harness for CUDA provider tests.
//!
//! Leaf modules retain their original `cuda`/`not(feature = "cuda")` gates;
//! one Cargo target replaces the previous flat target-per-file topology.

#[cfg(feature = "cuda")]
#[path = "cuda_ops/device/availability.rs"]
mod availability;
#[cfg(feature = "cuda")]
#[path = "cuda/cosine_similarity.rs"]
mod cosine_similarity;
#[path = "cuda_ops/device.rs"]
mod device;
#[cfg(feature = "cuda")]
#[path = "cuda/random_init.rs"]
mod random_init;
#[cfg(feature = "cuda")]
#[path = "cuda/rotate_half.rs"]
mod rotate_half;
#[path = "cuda_ops/unavailable.rs"]
mod unavailable;
