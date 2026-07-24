/// Batch normalization for 1D inputs `[N, C, L]`.
pub mod batchnorm1d;
/// Batch normalization for 2D inputs `[N, C, H, W]`.
pub mod batchnorm2d;
/// Batch normalization for 3D inputs `[N, C, D, H, W]`.
pub mod batchnorm3d;
/// Group normalization layer.
pub mod groupnorm;
/// Instance normalization layers (1D, 2D, 3D).
pub mod instancenorm;
/// Layer normalization (normalize over the last dimension).
pub mod layernorm;
/// RMS normalization layer.
pub mod rmsnorm;

pub use batchnorm1d::BatchNorm1d;
pub use batchnorm2d::BatchNorm2d;
pub use batchnorm3d::BatchNorm3d;
pub use groupnorm::{group_norm, GroupNorm};
pub use instancenorm::{InstanceNorm1d, InstanceNorm2d, InstanceNorm3d};
pub use layernorm::{layer_norm, LayerNorm};
pub use rmsnorm::{rms_norm, RMSNorm};
