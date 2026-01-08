/// Errors for `coeus-distributions`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A floating-point value was non-finite (NaN or ±∞).
    #[error("non-finite value for {name}: {value}")]
    NonFinite {
        /// Parameter name.
        name: &'static str,
        /// Rejected value.
        value: f64,
    },

    /// A numeric range was invalid.
    #[error("invalid range for {name}: min {min} is greater than max {max}")]
    InvalidRange {
        /// Distribution name.
        name: &'static str,
        /// Range lower bound.
        min: f64,
        /// Range upper bound.
        max: f64,
    },

    /// A parameter must be strictly positive.
    #[error("invalid positive value for {name}: {value}")]
    NonPositive {
        /// Parameter name.
        name: &'static str,
        /// Rejected value.
        value: f64,
    },

    /// A parameter must be non-negative.
    #[error("invalid non-negative value for {name}: {value}")]
    Negative {
        /// Parameter name.
        name: &'static str,
        /// Rejected value.
        value: f64,
    },

    /// A discrete distribution must have at least one value.
    #[error("empty support")]
    EmptySupport,

    /// Categorical distribution has mismatched category and weight lengths.
    #[error("categorical length mismatch: categories {categories_len}, weights {weights_len}")]
    CategoricalLengthMismatch {
        /// Category vector length.
        categories_len: usize,
        /// Weight vector length.
        weights_len: usize,
    },

    /// Categorical weights must sum to a strictly positive value.
    #[error("categorical weights must sum to a positive finite value")]
    CategoricalZeroTotalWeight,

    /// Categorical weights were rejected by `WeightedIndex`.
    #[error(transparent)]
    WeightedIndex(#[from] rand::distributions::weighted::WeightedError),

    /// Normal distribution parameters were rejected by `rand_distr`.
    #[error("invalid normal parameters: mean {mean}, std {std}")]
    InvalidNormalParameters {
        /// Mean parameter.
        mean: f64,
        /// Standard deviation parameter.
        std: f64,
    },

    /// A sample could not be represented as a JSON number.
    #[error("cannot represent sample as JSON number: {value}")]
    NonJsonNumber {
        /// Rejected value.
        value: f64,
    },

    /// A sampled value was outside its distribution support due to numerical issues.
    #[error("sample out of support for {name}: {value}")]
    SampleOutOfSupport {
        /// Distribution name.
        name: &'static str,
        /// Sampled value.
        value: f64,
    },
}

/// Result alias for `coeus-distributions`.
pub type Result<T> = core::result::Result<T, Error>;
