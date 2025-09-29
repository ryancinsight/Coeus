//! Normalization layers
//!
//! This module provides batch normalization, layer normalization,
//! and other normalization techniques for improving training stability.
//!
//! ## Mathematical Foundation
//!
//! ### Batch Normalization
//! ```math
//! μ_B = (1/m) * Σ xᵢ
//! σ_B² = (1/m) * Σ (xᵢ - μ_B)²
//! x̂ᵢ = (xᵢ - μ_B) / √(σ_B² + ε)
//! yᵢ = γ * x̂ᵢ + β
//! ```
//!
//! ## References
//!
//! - [Ioffe & Szegedy, 2015 - Batch Normalization](https://arxiv.org/abs/1502.03167)

// No direct imports needed - all functionality is re-exported from modular modules


