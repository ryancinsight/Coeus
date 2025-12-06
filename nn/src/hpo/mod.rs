//! Automated Hyperparameter Optimization (HPO).
//!
//! This module provides comprehensive hyperparameter optimization capabilities
//! including Bayesian optimization, multi-armed bandits, and population-based methods.

pub mod bandits;
pub mod bayesian;
pub mod multifidelity;
pub mod optimizer;
pub mod population;
pub mod space;

// Re-export main HPO types
pub use bandits::BanditOptimizer;
pub use bayesian::BayesianOptimizer;
pub use multifidelity::{HyperbandOptimizer, SuccessiveHalving};
pub use optimizer::{HPOptimizer, HyperparameterOptimizer, OptimizationResult};
pub use population::PopulationOptimizer;
pub use space::{HyperparameterConfig, HyperparameterSpace, HyperparameterValue};
