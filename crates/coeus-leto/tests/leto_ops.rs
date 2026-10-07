#![expect(
    clippy::unwrap_used,
    reason = "test assertions surface failures immediately by design"
)]

#[path = "leto_ops/contract/mod.rs"]
mod contract;
