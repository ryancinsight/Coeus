//! Provider-dispatched neural-network parameter initialization.

mod constant;
mod error;
mod random;

pub use constant::{constant, ones, zeros};
pub use error::InitializationError;
pub use random::{
    kaiming_normal, kaiming_normal_with_seed, kaiming_uniform, kaiming_uniform_with_seed, normal,
    normal_with_seed, uniform, uniform_with_seed, xavier_normal, xavier_normal_with_seed,
    xavier_uniform, xavier_uniform_with_seed,
};

/// Minimum tensor rank supported by provider-owned random initialization.
pub const MIN_INITIALIZER_RANK: usize = 1;
/// Maximum tensor rank supported by provider-owned random initialization.
pub const MAX_INITIALIZER_RANK: usize = coeus_leto::MAX_DISPATCH_RANK;
