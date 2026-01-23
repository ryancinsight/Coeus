//! Spatial dropout modules.

#[path = "2d.rs"]
pub mod spatial2d;
#[path = "3d.rs"]
pub mod spatial3d;

pub use spatial2d::Dropout2d;
pub use spatial3d::Dropout3d;
