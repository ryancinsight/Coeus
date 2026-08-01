//! Nonlinear least-squares fitting.
//!
//! The optimizers elsewhere in this crate step on gradients accumulated into
//! parameters, which is the shape network training takes. Fitting a model to
//! measurements is a different problem: the residual vector is the primitive,
//! its Jacobian exposes the Gauss-Newton curvature approximation `JᵀJ ≈ H`
//! that a bare gradient hides, and progress is decided by re-evaluating the
//! model at trial points.
//!
//! This module owns that shape. Define a [`LeastSquaresProblem`] and solve it
//! with [`levenberg_marquardt`].

mod config;
mod levenberg_marquardt;
mod problem;

#[cfg(test)]
mod tests;

pub use config::{LeastSquaresReport, LevenbergMarquardtConfig, Termination};
pub use levenberg_marquardt::{levenberg_marquardt, LeastSquaresScalar, SolverError};
pub use problem::{LeastSquaresProblem, ProblemError};
