//! Convolutional neural network layers
//!
//! This module provides 1D, 2D, and 3D convolutional layers
//! with various padding and stride options.
//!
//! ## Organization
//!
//! This module has been refactored for better modularity:
//! - `Conv1d` and `ConvTranspose1d` are in separate modules
//! - `Conv2d` and `ConvTranspose2d` are in separate modules
//! - `Conv3d` and `ConvTranspose3d` are in separate modules
//!
//! ## Mathematical Foundation
//!
//! ### 2D Convolution
//! ```math
//! (O[i,j,k]) = ΣᵤΣᵥ Σₘ (I[i+u, j+v, m] * W[u,v,m,k]) + B[k]
//!
//! Where:
//! - I: Input tensor of shape (batch_size, height, width, in_channels)
//! - W: Weight tensor of shape (kernel_height, kernel_width, in_channels, out_channels)
//! - B: Bias tensor of shape (out_channels,)
//! - O: Output tensor of shape (batch_size, out_height, out_width, out_channels)
//!
//! Output dimensions:
//! - out_height = (height + 2*padding_height - kernel_height) / stride_height + 1
//! - out_width = (width + 2*padding_width - kernel_width) / stride_width + 1
//! ```
//!
//! ## References
//!
//! - [Deep Learning Book - Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
//! - [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)

pub use crate::modules::conv1d::Conv1d;
pub use crate::modules::conv2d::Conv2d;
pub use crate::modules::conv_transpose2d::ConvTranspose2d;

// TODO: Add remaining convolutional layers (Conv3d, ConvTranspose1d, ConvTranspose3d) in separate modules

