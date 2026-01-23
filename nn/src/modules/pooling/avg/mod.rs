#[path = "1d.rs"]
pub mod avg1d;
#[path = "2d.rs"]
pub mod avg2d;
#[path = "3d.rs"]
pub mod avg3d;

pub use avg1d::AvgPool1d;
pub use avg2d::AvgPool2d;
pub use avg3d::AvgPool3d;
