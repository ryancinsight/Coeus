//! Neural-network module contracts and typed failures.

mod error;
mod trait_def;

pub use error::{ModuleError, ModuleResult, ParameterLoadError};
pub(crate) use trait_def::prefixed_parameters;
pub use trait_def::Module;
