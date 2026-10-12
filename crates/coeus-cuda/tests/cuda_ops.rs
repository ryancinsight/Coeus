//! Feature-aware integration harness for CUDA provider tests.
//!
//! Leaf modules retain their original `cuda`/`not(feature = "cuda")` gates;
//! one Cargo target replaces the previous flat target-per-file topology.

#[cfg(feature = "cuda")]
#[path = "cuda_ops/backend_writes.rs"]
mod backend_writes;

#[cfg(feature = "cuda")]
#[path = "cuda_ops/device_outputs.rs"]
mod device_outputs;

#[cfg(feature = "cuda")]
#[path = "cuda_ops/state_updates.rs"]
mod state_updates;

#[cfg(feature = "cuda")]
#[path = "cuda/attention.rs"]
mod attention;
#[cfg(feature = "cuda")]
#[path = "cuda_ops/device/availability.rs"]
mod availability;
#[cfg(feature = "cuda")]
#[path = "cuda/backend_ops.rs"]
mod backend_ops;
#[cfg(feature = "cuda")]
#[path = "cuda/conv_reduce.rs"]
mod conv_reduce;
#[cfg(feature = "cuda")]
#[path = "cuda/cosine_similarity.rs"]
mod cosine_similarity;
#[path = "cuda_ops/device.rs"]
mod device;
#[cfg(feature = "cuda")]
#[path = "cuda/jit.rs"]
mod jit;
#[cfg(feature = "cuda")]
#[path = "cuda/optimizer_and_3d.rs"]
mod optimizer_and_3d;
#[cfg(feature = "cuda")]
#[path = "cuda/parity.rs"]
mod parity;
#[cfg(feature = "cuda")]
#[path = "cuda/random_init.rs"]
mod random_init;
#[cfg(feature = "cuda")]
#[path = "cuda/rotate_half.rs"]
mod rotate_half;
#[cfg(feature = "cuda")]
#[path = "cuda/strided.rs"]
mod strided;
#[path = "cuda_ops/unavailable.rs"]
mod unavailable;
