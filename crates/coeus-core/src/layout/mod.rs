// ── Layout module ──
// Shape, strides, and multi-dimensional layout descriptors.

#[expect(clippy::module_inception, reason = "ratchet COEUS-LINT-1")]
mod layout;
mod shape;
mod strides;

pub use layout::{ConstLayout, Layout};
pub use shape::{ConstShape, Shape};
pub use strides::{is_contiguous, row_major_strides, Strides};
