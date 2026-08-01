//! Generic provider bridge for in-place stateful parameter updates.

mod dispatch;
mod implementation;
mod provider;

pub use provider::{StatefulUpdateBackend, StatefulUpdateProvider};
