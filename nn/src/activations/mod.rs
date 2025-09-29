//! Activation functions for neural networks
//!
//! This module provides common activation functions used in neural networks.
//! All activations implement the `Module` trait for seamless integration.
//!
//! ## Available Activations
//!
//! - **ReLU**: Rectified Linear Unit, `max(0, x)`
//! - **Sigmoid**: Logistic function, `1 / (1 + exp(-x))`
//! - **Tanh**: Hyperbolic tangent, `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
//! - **Softmax**: Normalized exponential function
//! - **LeakyReLU**: Leaky version of ReLU, `max(αx, x)`
//!
//! ## Mathematical Properties
//!
//! ### ReLU
//! ```math
//! ReLU(x) = max(0, x)
//!
//! ∂ReLU/∂x = {
//!     1  if x > 0
//!     0  if x ≤ 0
//! }
//! ```
//!
//! ### Sigmoid
//! ```math
//! σ(x) = 1 / (1 + exp(-x))
//!
//! ∂σ/∂x = σ(x) * (1 - σ(x))
//! ```
//!
//! ## References
//!
//! - [Deep Learning Book - Activation Functions](https://www.deeplearningbook.org/contents/mlp.html)
//! - [Glorot & Bengio, 2010 - Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
//! - [Hendrycks & Gimpel, 2016 - Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)

pub mod elementwise;
pub mod sigmoid_family;
pub mod hyperbolic;
pub mod softmax_family;

// Re-export all activation functions for backward compatibility
pub use elementwise::*;
pub use sigmoid_family::*;
pub use hyperbolic::*;
pub use softmax_family::*;


