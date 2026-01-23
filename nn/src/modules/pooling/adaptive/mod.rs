pub mod avg1d;
pub mod avg2d;
pub mod avg3d;
pub mod max1d;
pub mod max2d;
pub mod max3d;

pub use avg1d::AdaptiveAvgPool1d;
pub use avg2d::AdaptiveAvgPool2d;
pub use avg3d::AdaptiveAvgPool3d;
pub use max1d::AdaptiveMaxPool1d;
pub use max2d::AdaptiveMaxPool2d;
pub use max3d::AdaptiveMaxPool3d;
