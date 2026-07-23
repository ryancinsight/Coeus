//! Feature-aware integration harness for CUDA device and fallback tests.
//!
//! Leaf modules retain their original `cuda`/`not(feature = "cuda")` gates;
//! one Cargo target replaces the previous flat target-per-file topology.

#[path = "cuda_ops/device.rs"]
mod device;
#[path = "cuda_ops/fallback.rs"]
mod fallback;
