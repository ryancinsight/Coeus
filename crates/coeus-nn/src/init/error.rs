use std::error::Error;

/// Failure returned by neural-network random initialization.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InitializationError<E>
where
    E: Error + 'static,
{
    /// The tensor rank is outside the shared provider dispatch domain.
    #[error("random initialization requires rank {minimum}..={maximum}, got {actual}")]
    InvalidRank {
        /// Observed tensor rank.
        actual: usize,
        /// Smallest supported rank.
        minimum: usize,
        /// Largest supported rank.
        maximum: usize,
    },
    /// A distribution parameter is not finite in the requested scalar type.
    #[error("initialization parameter {parameter} must be finite, got {value}")]
    NonFiniteParameter {
        /// Rejected parameter name.
        parameter: &'static str,
        /// Original public input value.
        value: f64,
    },
    /// Uniform bounds are reversed.
    #[error("uniform lower bound must not exceed upper bound, got {low} > {high}")]
    InvalidUniformBounds {
        /// Requested lower bound.
        low: f64,
        /// Requested upper bound.
        high: f64,
    },
    /// A normal standard deviation is negative.
    #[error("normal standard deviation must be non-negative, got {value}")]
    NegativeStandardDeviation {
        /// Rejected standard deviation.
        value: f64,
    },
    /// A fan parameter is zero.
    #[error("{parameter} must be positive, got {value}")]
    InvalidFan {
        /// Rejected fan parameter.
        parameter: &'static str,
        /// Rejected value.
        value: usize,
    },
    /// Attention width cannot be partitioned across the configured heads.
    #[error("attention d_model {d_model} must be positive and divisible by {heads} heads")]
    InvalidHeadConfiguration {
        /// Model width.
        d_model: usize,
        /// Compile-time head count.
        heads: usize,
    },
    /// A transformer stack requested no layers.
    #[error("transformer layer count must be positive, got {layers}")]
    InvalidLayerCount {
        /// Compile-time layer count.
        layers: usize,
    },
    /// Xavier fan arithmetic overflowed `usize`.
    #[error("fan_in + fan_out exceeds usize: {fan_in} + {fan_out}")]
    FanArithmeticOverflow {
        /// Input fan.
        fan_in: usize,
        /// Output fan.
        fan_out: usize,
    },
    /// A derived fan product overflowed `usize`.
    #[error("{lhs_name} * {rhs_name} exceeds usize: {lhs} * {rhs}")]
    FanProductOverflow {
        /// Left factor name.
        lhs_name: &'static str,
        /// Left factor value.
        lhs: usize,
        /// Right factor name.
        rhs_name: &'static str,
        /// Right factor value.
        rhs: usize,
    },
    /// The selected backend provider rejected initialization.
    #[error("{operation} backend initialization failed")]
    Backend {
        /// Initialization family being dispatched.
        operation: &'static str,
        /// Backend-owned typed failure.
        #[source]
        source: E,
    },
}
