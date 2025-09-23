//! Pooling layers for neural networks (Legacy monolithic module)
//!
//! **DEPRECATED**: This module contains the original monolithic pooling implementations.
//! New code should use the modular pooling modules:
//! - `maxpool1d`, `maxpool2d`, `maxpool3d` for max pooling
//! - `avgpool1d`, `avgpool2d`, `avgpool3d` for average pooling
//! - `adaptive_maxpool1d`, `adaptive_maxpool2d`, `adaptive_maxpool3d` for adaptive max pooling
//! - `adaptive_avgpool1d`, `adaptive_avgpool2d`, `adaptive_avgpool3d` for adaptive average pooling
//!
//! ## Mathematical Foundation
//!
//! ### Max Pooling
//! ```math
//! O[i,j,k] = max_{u,v} I[i*stride+u, j*stride+v, k]
//! ```
//!
//! ### Average Pooling
//! ```math
//! O[i,j,k] = (1/(kernel_h * kernel_w)) * Σ_{u,v} I[i*stride+u, j*stride+v, k]
//! ```
//!
//! ## References
//!
//! - [Deep Learning Book - Pooling](https://www.deeplearningbook.org/contents/convnets.html)
//! - [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)

// No direct imports needed - all functionality is re-exported from modular modules

// Re-export legacy monolithic implementations for backward compatibility
// These will be removed in a future version

// Re-export the modular implementations for cleaner API
pub use crate::modules::maxpool1d::MaxPool1d;
pub use crate::modules::maxpool2d::MaxPool2d;
pub use crate::modules::maxpool3d::MaxPool3d;
pub use crate::modules::avgpool1d::AvgPool1d;
pub use crate::modules::avgpool2d::AvgPool2d;
pub use crate::modules::avgpool3d::AvgPool3d;
pub use crate::modules::adaptive_maxpool1d::AdaptiveMaxPool1d;
pub use crate::modules::adaptive_maxpool2d::AdaptiveMaxPool2d;
pub use crate::modules::adaptive_maxpool3d::AdaptiveMaxPool3d;
pub use crate::modules::adaptive_avgpool1d::AdaptiveAvgPool1d;
pub use crate::modules::adaptive_avgpool2d::AdaptiveAvgPool2d;
pub use crate::modules::adaptive_avgpool3d::AdaptiveAvgPool3d;