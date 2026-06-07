// ── Layout module ──
// Shape, strides, and multi-dimensional layout descriptors.

mod shape;
mod strides;
#[allow(clippy::module_inception)]
mod layout;

pub use shape::{Shape, ConstShape};
pub use strides::Strides;
pub use layout::{Layout, ConstLayout};
