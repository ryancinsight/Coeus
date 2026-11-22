//! Reverse-mode derivatives of homogeneous physical quantities in canonical SI.
//!
//! Use Aequitas for unit conversion and dimension algebra. A scoped graph
//! associates each gradient with one scalar objective and one input field.
//! Represent a scalar as a one-element tensor, for example with shape `[1]`.
//!
//! Incompatible addition fails at the type boundary.
//!
//! ```compile_fail,E0277
//! use aequitas::systems::si::dimensions::{Length, Time};
//! use coeus_autograd::physical::{add, Variable};
//! use coeus_core::MoiraiBackend;
//! fn invalid<'s>(
//!     length: &Variable<'s, f64, MoiraiBackend, Length>,
//!     time: &Variable<'s, f64, MoiraiBackend, Time>,
//! ) {
//!     let _ = add(length, time);
//! }
//! ```
//!
//! Absolute temperatures can be subtracted to form an interval.
//!
//! ```compile_fail,E0277
//! use aequitas::systems::si::dimensions::ThermodynamicTemperature;
//! use coeus_autograd::physical::{add, Variable};
//! use coeus_core::MoiraiBackend;
//! fn invalid<'s>(point: &Variable<'s, f64, MoiraiBackend, ThermodynamicTemperature>) {
//!     let _ = add(point, point);
//! }
//! ```
//!
//! An interval requires a linear degree, and a logarithm requires a
//! dimensionless ratio.
//!
//! ```compile_fail,E0277
//! use aequitas::systems::si::{dimensions::TemperatureDifference, units::DegreeFahrenheit};
//! use coeus_autograd::physical::Coordinates;
//! let _ = Coordinates::<TemperatureDifference>::affine::<DegreeFahrenheit>();
//! ```
//!
//! ```compile_fail,E0308
//! use aequitas::systems::si::dimensions::Length;
//! use coeus_autograd::physical::{log, Variable};
//! use coeus_core::MoiraiBackend;
//! fn invalid<'s>(length: &Variable<'s, f64, MoiraiBackend, Length>) {
//!     let _ = log(length);
//! }
//! ```
//!
//! Scoped variables cannot be relabelled or carried into another graph.
//!
//! ```compile_fail
//! use aequitas::systems::si::dimensions::Length;
//! use coeus_autograd::physical::{value_and_gradient, Variable};
//! use coeus_core::MoiraiBackend;
//! use coeus_tensor::Tensor;
//! use aequitas::Quantity;
//! let x = Quantity::<_, Length>::from_base(
//!     Tensor::<f64, MoiraiBackend>::from_slice([1], &[2.0]).unwrap(),
//! );
//! let mut escaped: Option<Variable<'_, f64, MoiraiBackend, Length>> = None;
//! let _ = value_and_gradient(x, |scope, input| {
//!     escaped = Some(input);
//!     scope.constant(Quantity::<_, Length>::from_base(
//!         Tensor::from_slice([1], &[1.0]).unwrap(),
//!     ))
//! });
//! drop(escaped);
//! ```

mod coordinates;
mod graph;
mod operations;

pub use coordinates::Coordinates;
pub use graph::{value_and_gradient, Gradient, QuantityTensor, Scope, ValueAndGradient, Variable};
pub use operations::{add, div, log, mul, sub, sum};

/// Failure to compute or express a physical derivative.
#[derive(Debug, thiserror::Error)]
pub enum Error<E> {
    /// The numerical backend failed.
    #[error(transparent)]
    Backend(E),
    /// Rank-zero tensors are outside the physical adapter's shape contract.
    #[error("physical differentiation expects an explicit tensor axis (use [1] for a scalar)")]
    RankZero,
    /// The output must contain one element.
    #[error("a physical gradient requires a one-element objective, got {elements} elements")]
    NonScalarObjective {
        /// Element count of the objective.
        elements: usize,
    },
    /// Recording is disabled on the current thread.
    #[error("physical differentiation requires gradient recording")]
    RecordingDisabled,
    /// The tracked input's gradient buffer could not be read.
    #[error("tracked physical input is missing its gradient buffer")]
    MissingGradient,
    /// The coordinate scale or offset is outside the scalar's range.
    #[error("coordinate scale or offset is not representable")]
    CoordinateRange,
}
