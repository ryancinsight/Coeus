//! Hierarchical integration harness for core storage, policy, and scalar tests.
//!
//! The leaf modules retain their original contract assertions; one Cargo target
//! replaces the previous flat target-per-file topology. Library unit modules
//! remain in `src` and are intentionally outside this integration harness.

#[path = "core_ops/policy.rs"]
mod policy;
#[path = "core_ops/scalars.rs"]
mod scalars;
#[path = "core_ops/storage.rs"]
mod storage;
