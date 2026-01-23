pub mod adaptive;
pub mod avg;
pub mod core;
pub mod max;

pub use adaptive::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool1d, AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
};
pub use avg::{AvgPool1d, AvgPool2d, AvgPool3d};
pub use max::{MaxPool1d, MaxPool2d, MaxPool3d};
