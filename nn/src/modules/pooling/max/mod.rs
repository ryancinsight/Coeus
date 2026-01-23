#[path = "1d.rs"]
pub mod max1d;
#[path = "2d.rs"]
pub mod max2d;
#[path = "3d.rs"]
pub mod max3d;

pub use max1d::MaxPool1d;
pub use max2d::MaxPool2d;
pub use max3d::MaxPool3d;
