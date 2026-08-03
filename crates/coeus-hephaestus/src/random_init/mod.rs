//! Generic provider-owned random initialization.

mod dispatch;
mod provider;

pub use dispatch::{normal, uniform};
pub use provider::RandomInitProvider;
