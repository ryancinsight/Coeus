pub mod bilinear;
pub mod dense;
pub mod lazy;
pub mod sparse;

pub use bilinear::Bilinear;

pub use dense::Linear;
pub use lazy::LazyLinear;
pub use sparse::SparseLinear;
