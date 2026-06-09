// ── Layout module ──
// Shape, strides, and multi-dimensional layout descriptors.

#[allow(clippy::module_inception)]
mod layout;
mod shape;
mod strides;

pub use layout::{ConstLayout, Layout};
pub use shape::{ConstShape, Shape};
pub use strides::Strides;
