#![allow(clippy::too_many_arguments)]

mod contiguous;
mod strided;

pub use contiguous::{launch_contiguous_binary, launch_contiguous_unary};
pub use strided::{launch_strided_binary, launch_strided_unary};
