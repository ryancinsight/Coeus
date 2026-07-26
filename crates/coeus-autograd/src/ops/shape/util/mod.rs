mod contiguous;
mod cumprod;
mod cumsum;
mod diff;
mod einsum;

pub use contiguous::contiguous;
pub use cumprod::cumprod;
pub use cumsum::cumsum;
pub use diff::diff;
pub use einsum::{einsum, einsum3};
