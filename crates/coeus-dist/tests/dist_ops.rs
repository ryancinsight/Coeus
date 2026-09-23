//! Hierarchical distributed-contract integration harness.

#![expect(
    clippy::unwrap_used,
    reason = "test/bench/example code asserts and reports by design"
)]

#[path = "distributed/local/mod.rs"]
mod local;
#[path = "distributed/support.rs"]
mod support;
#[path = "distributed/tcp/mod.rs"]
mod tcp;
