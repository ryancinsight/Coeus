//! Typed gradcheck failures.

/// Why a [`fn@gradcheck`] run did not establish agreement.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum GradcheckError {
    /// The loss closure returned a non-scalar. A gradcheck needs one scalar
    /// output so that a single backward pass yields the full gradient.
    NonScalarLoss {
        /// Shape the closure actually returned.
        shape: Vec<usize>,
    },
    /// The reverse pass left an input without a gradient, so there is nothing
    /// to compare. Usually the input was not reached by the graph the closure
    /// built.
    MissingGradient {
        /// Index into the `inputs` slice.
        input: usize,
    },
    /// Both gradients are indistinguishable from zero, so the comparison has no
    /// discriminating power. See the module-level zero-gradient guard.
    TriviallyZero {
        /// Largest absolute analytic gradient component observed.
        max_analytic: f64,
        /// Largest absolute numeric gradient component observed.
        max_numeric: f64,
        /// Magnitude below which a gradient counts as zero.
        floor: f64,
    },
    /// An analytic component disagrees with the finite-difference estimate by
    /// more than the derived tolerance.
    Mismatch {
        /// Index into the `inputs` slice.
        input: usize,
        /// Flat element index within that input.
        element: usize,
        /// Component reported by `backward`.
        analytic: f64,
        /// Component reconstructed from forward evaluations.
        numeric: f64,
        /// Largest difference that would have passed.
        tolerance: f64,
    },
    /// The backward pass itself failed.
    Backward(String),
}

impl core::fmt::Display for GradcheckError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonScalarLoss { shape } => write!(
                f,
                "gradcheck requires a scalar loss; closure returned shape {shape:?}"
            ),
            Self::MissingGradient { input } => write!(
                f,
                "input {input} received no gradient; it is not reachable from the loss"
            ),
            Self::TriviallyZero {
                max_analytic,
                max_numeric,
                floor,
            } => write!(
                f,
                "gradcheck is vacuous: analytic ({max_analytic:.3e}) and numeric \
                 ({max_numeric:.3e}) gradients are both below the zero floor {floor:.3e}. \
                 Weight the loss non-uniformly so the Jacobian is actually probed."
            ),
            Self::Mismatch {
                input,
                element,
                analytic,
                numeric,
                tolerance,
            } => write!(
                f,
                "gradient mismatch at input {input} element {element}: analytic {analytic:.9e} \
                 vs numeric {numeric:.9e} (difference {:.3e} exceeds tolerance {tolerance:.3e})",
                (analytic - numeric).abs()
            ),
            Self::Backward(message) => write!(f, "backward pass failed: {message}"),
        }
    }
}

impl std::error::Error for GradcheckError {}
