use rand::distributions::WeightedIndex;
use rand::Rng;
use rand_distr::Distribution as _;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A finite `f64` (not NaN, not ±∞).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FiniteF64(f64);

impl FiniteF64 {
    /// Create a finite `f64`.
    ///
    /// # Errors
    /// Returns [`Error::NonFinite`] if `value` is NaN or infinite.
    pub fn new(name: &'static str, value: f64) -> Result<Self> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(Error::NonFinite { name, value })
        }
    }

    /// Get the underlying value.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A strictly positive, finite `f64`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PositiveFiniteF64(f64);

impl PositiveFiniteF64 {
    /// Create a strictly positive, finite `f64`.
    ///
    /// # Errors
    /// Returns [`Error::NonFinite`] if `value` is NaN or infinite, or
    /// [`Error::NonPositive`] if `value <= 0`.
    pub fn new(name: &'static str, value: f64) -> Result<Self> {
        let finite = FiniteF64::new(name, value)?;
        if finite.get() > 0.0 {
            Ok(Self(finite.get()))
        } else {
            Err(Error::NonPositive {
                name,
                value: finite.get(),
            })
        }
    }

    /// Get the underlying value.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A non-negative, finite `f64`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NonNegativeFiniteF64(f64);

impl NonNegativeFiniteF64 {
    /// Create a non-negative, finite `f64`.
    ///
    /// # Errors
    /// Returns [`Error::NonFinite`] if `value` is NaN or infinite, or
    /// [`Error::Negative`] if `value < 0`.
    pub fn new(name: &'static str, value: f64) -> Result<Self> {
        let finite = FiniteF64::new(name, value)?;
        if finite.get() >= 0.0 {
            Ok(Self(finite.get()))
        } else {
            Err(Error::Negative {
                name,
                value: finite.get(),
            })
        }
    }

    /// Get the underlying value.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
enum ParameterDistributionRepr {
    Continuous {
        min: f64,
        max: f64,
    },
    Normal {
        mean: f64,
        std: f64,
    },
    Logarithmic {
        min: f64,
        max: f64,
        base: f64,
    },
    Discrete {
        values: Vec<serde_json::Value>,
    },
    Categorical {
        categories: Vec<String>,
        weights: Vec<f64>,
    },
}

/// A validated distribution specification for parameter sampling.
///
/// Invariants:
/// - All numeric parameters are finite.
/// - `Continuous`: `min <= max`.
/// - `Normal`: `std > 0`.
/// - `Logarithmic`: `min <= max`, `base > 0`.
/// - `Discrete`: `values` is non-empty.
/// - `Categorical`: `categories` is non-empty, `categories.len() == weights.len()`,
///   all weights are finite and non-negative, and the total weight is positive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(
    try_from = "ParameterDistributionRepr",
    into = "ParameterDistributionRepr"
)]
pub enum ParameterDistribution {
    /// Continuous uniform distribution over `[min, max]`.
    Continuous {
        /// Range lower bound.
        min: FiniteF64,
        /// Range upper bound.
        max: FiniteF64,
    },
    /// Normal distribution.
    Normal {
        /// Mean parameter.
        mean: FiniteF64,
        /// Standard deviation parameter.
        std: PositiveFiniteF64,
    },
    /// Sample `v ~ Uniform([min, max])` then return `base^v`.
    Logarithmic {
        /// Exponent range lower bound.
        min: FiniteF64,
        /// Exponent range upper bound.
        max: FiniteF64,
        /// Exponentiation base.
        base: PositiveFiniteF64,
    },
    /// Discrete set of JSON values.
    Discrete {
        /// Support values.
        values: Vec<serde_json::Value>,
    },
    /// Categorical distribution over string categories with non-negative weights.
    Categorical {
        /// Category labels.
        categories: Vec<String>,
        /// Non-negative category weights.
        weights: Vec<NonNegativeFiniteF64>,
    },
}

impl ParameterDistribution {
    /// Create a validated continuous uniform distribution.
    ///
    /// # Errors
    /// Returns [`Error::NonFinite`] if either bound is non-finite, or
    /// [`Error::InvalidRange`] if `min > max`.
    pub fn continuous(min: f64, max: f64) -> Result<Self> {
        Self::try_from(ParameterDistributionRepr::Continuous { min, max })
    }

    /// Create a validated normal distribution.
    ///
    /// # Errors
    /// Returns [`Error::NonFinite`] if `mean` or `std` is non-finite, or
    /// [`Error::NonPositive`] if `std <= 0`.
    pub fn normal(mean: f64, std: f64) -> Result<Self> {
        Self::try_from(ParameterDistributionRepr::Normal { mean, std })
    }

    /// Create a validated logarithmic distribution.
    ///
    /// # Errors
    /// Returns [`Error::NonFinite`] if any argument is non-finite, or
    /// [`Error::InvalidRange`] if `min > max`, or
    /// [`Error::NonPositive`] if `base <= 0`.
    pub fn logarithmic(min: f64, max: f64, base: f64) -> Result<Self> {
        Self::try_from(ParameterDistributionRepr::Logarithmic { min, max, base })
    }

    /// Create a validated discrete distribution.
    ///
    /// # Errors
    /// Returns [`Error::EmptySupport`] if `values` is empty.
    pub fn discrete(values: Vec<serde_json::Value>) -> Result<Self> {
        Self::try_from(ParameterDistributionRepr::Discrete { values })
    }

    /// Create a validated categorical distribution.
    ///
    /// # Errors
    /// Returns [`Error::EmptySupport`] if `categories` is empty,
    /// [`Error::CategoricalLengthMismatch`] if `categories.len() != weights.len()`,
    /// [`Error::NonFinite`] if any weight is non-finite,
    /// [`Error::Negative`] if any weight is negative, or
    /// [`Error::CategoricalZeroTotalWeight`] if the total weight is not positive.
    pub fn categorical(categories: Vec<String>, weights: Vec<f64>) -> Result<Self> {
        Self::try_from(ParameterDistributionRepr::Categorical {
            categories,
            weights,
        })
    }

    /// Sample a value and return it as `serde_json::Value`.
    ///
    /// # Errors
    /// Returns an error if sampling fails due to invalid weights, invalid normal
    /// parameters, or if a sampled numeric value cannot be represented as a JSON
    /// number.
    pub fn sample_json<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<serde_json::Value> {
        match self {
            Self::Continuous { min, max } => {
                let x = rng.gen_range(min.get()..=max.get());
                serde_json::Number::from_f64(x)
                    .map(serde_json::Value::Number)
                    .ok_or(Error::NonJsonNumber { value: x })
            }
            Self::Normal { mean, std } => {
                let dist = rand_distr::Normal::new(mean.get(), std.get()).map_err(|_| {
                    Error::InvalidNormalParameters {
                        mean: mean.get(),
                        std: std.get(),
                    }
                })?;
                let x = dist.sample(rng);
                if !x.is_finite() {
                    return Err(Error::SampleOutOfSupport {
                        name: "Normal",
                        value: x,
                    });
                }
                serde_json::Number::from_f64(x)
                    .map(serde_json::Value::Number)
                    .ok_or(Error::NonJsonNumber { value: x })
            }
            Self::Logarithmic { min, max, base } => {
                let v = rng.gen_range(min.get()..=max.get());
                let x = base.get().powf(v);
                if !x.is_finite() {
                    return Err(Error::SampleOutOfSupport {
                        name: "Logarithmic",
                        value: x,
                    });
                }
                serde_json::Number::from_f64(x)
                    .map(serde_json::Value::Number)
                    .ok_or(Error::NonJsonNumber { value: x })
            }
            Self::Discrete { values } => {
                if values.is_empty() {
                    return Err(Error::EmptySupport);
                }
                let idx = rng.gen_range(0..values.len());
                Ok(values[idx].clone())
            }
            Self::Categorical {
                categories,
                weights,
            } => {
                if categories.is_empty() {
                    return Err(Error::EmptySupport);
                }
                if categories.len() != weights.len() {
                    return Err(Error::CategoricalLengthMismatch {
                        categories_len: categories.len(),
                        weights_len: weights.len(),
                    });
                }
                let w: Vec<f64> = weights.iter().map(|x| x.get()).collect();
                if w.iter().sum::<f64>() <= 0.0 {
                    return Err(Error::CategoricalZeroTotalWeight);
                }
                let dist = WeightedIndex::new(w)?;
                let idx = dist.sample(rng);
                Ok(serde_json::Value::String(categories[idx].clone()))
            }
        }
    }
}

impl TryFrom<ParameterDistributionRepr> for ParameterDistribution {
    type Error = Error;

    fn try_from(value: ParameterDistributionRepr) -> Result<Self> {
        match value {
            ParameterDistributionRepr::Continuous { min, max } => {
                let min = FiniteF64::new("min", min)?;
                let max = FiniteF64::new("max", max)?;
                if min.get() > max.get() {
                    return Err(Error::InvalidRange {
                        name: "Continuous",
                        min: min.get(),
                        max: max.get(),
                    });
                }
                Ok(Self::Continuous { min, max })
            }
            ParameterDistributionRepr::Normal { mean, std } => {
                let mean = FiniteF64::new("mean", mean)?;
                let std = PositiveFiniteF64::new("std", std)?;
                Ok(Self::Normal { mean, std })
            }
            ParameterDistributionRepr::Logarithmic { min, max, base } => {
                let min = FiniteF64::new("min", min)?;
                let max = FiniteF64::new("max", max)?;
                if min.get() > max.get() {
                    return Err(Error::InvalidRange {
                        name: "Logarithmic",
                        min: min.get(),
                        max: max.get(),
                    });
                }
                let base = PositiveFiniteF64::new("base", base)?;
                Ok(Self::Logarithmic { min, max, base })
            }
            ParameterDistributionRepr::Discrete { values } => {
                if values.is_empty() {
                    return Err(Error::EmptySupport);
                }
                Ok(Self::Discrete { values })
            }
            ParameterDistributionRepr::Categorical {
                categories,
                weights,
            } => {
                if categories.is_empty() {
                    return Err(Error::EmptySupport);
                }
                if categories.len() != weights.len() {
                    return Err(Error::CategoricalLengthMismatch {
                        categories_len: categories.len(),
                        weights_len: weights.len(),
                    });
                }
                let weights: Vec<NonNegativeFiniteF64> = weights
                    .into_iter()
                    .map(|w| NonNegativeFiniteF64::new("weight", w))
                    .collect::<Result<_>>()?;
                if weights.iter().map(|w| w.get()).sum::<f64>() <= 0.0 {
                    return Err(Error::CategoricalZeroTotalWeight);
                }
                Ok(Self::Categorical {
                    categories,
                    weights,
                })
            }
        }
    }
}

impl From<ParameterDistribution> for ParameterDistributionRepr {
    fn from(value: ParameterDistribution) -> Self {
        match value {
            ParameterDistribution::Continuous { min, max } => Self::Continuous {
                min: min.get(),
                max: max.get(),
            },
            ParameterDistribution::Normal { mean, std } => Self::Normal {
                mean: mean.get(),
                std: std.get(),
            },
            ParameterDistribution::Logarithmic { min, max, base } => Self::Logarithmic {
                min: min.get(),
                max: max.get(),
                base: base.get(),
            },
            ParameterDistribution::Discrete { values } => Self::Discrete { values },
            ParameterDistribution::Categorical {
                categories,
                weights,
            } => Self::Categorical {
                categories,
                weights: weights.into_iter().map(NonNegativeFiniteF64::get).collect(),
            },
        }
    }
}
