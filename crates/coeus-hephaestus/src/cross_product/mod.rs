mod dispatch;
mod host_fallback;
mod provider;
#[cfg(test)]
mod tests;

pub use dispatch::cross_product;
pub use host_fallback::{cross_host_fallback, HostCrossProductOps};
pub use provider::CrossProductProvider;
