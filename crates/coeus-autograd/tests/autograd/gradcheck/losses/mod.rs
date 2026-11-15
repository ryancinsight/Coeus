//! Finite-difference checks for the loss backward passes.
//!
//! # Placing the inputs off the kinks
//!
//! Most of these losses are piecewise: `l1` and `smooth_l1` break where the
//! residual is `0` or `±beta`, the margin family breaks where the hinge
//! argument crosses `0`. A central difference straddles its evaluation point, so
//! at a break it averages the two one-sided derivatives and disagrees with any
//! correct backward — the check would report a defect that is not there.
//!
//! The residual-based losses therefore take an explicitly chosen residual
//! vector rather than a sampled one, so the distance from every break is a
//! stated number rather than an accident of the generator. The finite-difference
//! step is `ε^(1/3)·max(|x|,1) ≈ 6.1e-6` in `f64`, so a residual `0.1` away from
//! a break is four orders of magnitude clear of it: no perturbation can cross.
//!
//! The checks live in leaf modules by loss family: `regression` for the
//! residual-based losses, `classification` for the log-probability losses,
//! `ranking` for the margin losses and `similarity` for the distance and
//! embedding losses.

mod classification;
mod ranking;
mod regression;
mod similarity;

use super::{tensor, GradcheckScalar, Sampler};

/// Shape shared by the element-wise regression losses.
const PAIR_SHAPE: [usize; 2] = [2, 3];
