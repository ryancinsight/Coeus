//! Shape manipulation operations

pub mod broadcast;
pub mod cat;
pub mod flatten;
pub mod permute;
pub mod reshape;
pub mod squeeze;
pub mod transpose;
pub mod view;

pub mod unsqueeze;

// Re-exports
pub use broadcast::{broadcast_shapes, broadcast_tensor_data};
pub use cat::cat;
pub use flatten::flatten;
pub use permute::permute;
pub use reshape::reshape;
pub use squeeze::squeeze;
pub use transpose::transpose;
pub use view::view;
pub use unsqueeze::unsqueeze;
