use coeus_autograd::Var;
use coeus_core::{Float, Layout, Scalar};
use coeus_ops::RandomInitOps;
use coeus_tensor::Tensor;

use super::{InitializationError, MAX_INITIALIZER_RANK, MIN_INITIALIZER_RANK};

type InitResult<B> =
    std::result::Result<(), InitializationError<<B as coeus_core::ComputeBackend>::Error>>;

fn validate_layout<E: std::error::Error + 'static>(
    layout: &Layout,
) -> std::result::Result<(), InitializationError<E>> {
    let rank = layout.ndim();
    if (MIN_INITIALIZER_RANK..=MAX_INITIALIZER_RANK).contains(&rank) {
        Ok(())
    } else {
        Err(InitializationError::InvalidRank {
            actual: rank,
            minimum: MIN_INITIALIZER_RANK,
            maximum: MAX_INITIALIZER_RANK,
        })
    }
}

fn finite<T: Float, E: std::error::Error + 'static>(
    parameter: &'static str,
    value: f64,
) -> std::result::Result<T, InitializationError<E>> {
    let converted = <T as Scalar>::from_f64(value);
    if value.is_finite() && <T as Float>::is_finite(converted) {
        Ok(converted)
    } else {
        Err(InitializationError::NonFiniteParameter { parameter, value })
    }
}

fn dense_layout<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &Var<T, B>) -> Layout {
    Layout::new(weight.tensor.shape_cloned())
}

fn uniform_typed_with_seed<T, B>(
    weight: &mut Var<T, B>,
    low: T,
    high: T,
    seed: u64,
) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    let layout = dense_layout(weight);
    validate_layout(&layout)?;
    if !<T as Float>::is_finite(low) {
        return Err(InitializationError::NonFiniteParameter {
            parameter: "low",
            value: <T as Scalar>::to_f64(low),
        });
    }
    if !<T as Float>::is_finite(high) {
        return Err(InitializationError::NonFiniteParameter {
            parameter: "high",
            value: <T as Scalar>::to_f64(high),
        });
    }
    if low > high {
        return Err(InitializationError::InvalidUniformBounds {
            low: <T as Scalar>::to_f64(low),
            high: <T as Scalar>::to_f64(high),
        });
    }
    let storage = B::default()
        .uniform_random(&layout, low, high, seed)
        .map_err(|source| InitializationError::Backend {
            operation: "uniform",
            source,
        })?;
    weight.tensor = Tensor::from_raw_parts(storage, layout);
    Ok(())
}

fn normal_typed_with_seed<T, B>(
    weight: &mut Var<T, B>,
    mean: T,
    std_dev: T,
    seed: u64,
) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    let layout = dense_layout(weight);
    validate_layout(&layout)?;
    if !<T as Float>::is_finite(mean) {
        return Err(InitializationError::NonFiniteParameter {
            parameter: "mean",
            value: <T as Scalar>::to_f64(mean),
        });
    }
    if !<T as Float>::is_finite(std_dev) {
        return Err(InitializationError::NonFiniteParameter {
            parameter: "std_dev",
            value: <T as Scalar>::to_f64(std_dev),
        });
    }
    if std_dev < T::zero() {
        return Err(InitializationError::NegativeStandardDeviation {
            value: <T as Scalar>::to_f64(std_dev),
        });
    }
    let storage = B::default()
        .normal_random(&layout, mean, std_dev, seed)
        .map_err(|source| InitializationError::Backend {
            operation: "normal",
            source,
        })?;
    weight.tensor = Tensor::from_raw_parts(storage, layout);
    Ok(())
}

/// Initialize weights with values from a uniform distribution using a seed.
///
/// # Errors
///
/// Returns [`InitializationError`] for unsupported rank, non-finite or
/// reversed bounds, or a selected-provider failure. The original tensor is
/// unchanged on error.
pub fn uniform_with_seed<T, B>(
    weight: &mut Var<T, B>,
    low: f64,
    high: f64,
    seed: u64,
) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    let low_t = finite::<T, B::Error>("low", low)?;
    let high_t = finite::<T, B::Error>("high", high)?;
    if low > high {
        return Err(InitializationError::InvalidUniformBounds { low, high });
    }
    uniform_typed_with_seed(weight, low_t, high_t, seed)
}

/// Initialize weights with values from a uniform distribution using seed 42.
///
/// # Errors
///
/// Returns the same failures as [`uniform_with_seed`].
pub fn uniform<T, B>(weight: &mut Var<T, B>, low: f64, high: f64) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    uniform_with_seed(weight, low, high, 42)
}

/// Initialize weights with values from a normal distribution using a seed.
///
/// # Errors
///
/// Returns [`InitializationError`] for unsupported rank, non-finite parameters,
/// negative standard deviation, or a selected-provider failure. The original
/// tensor is unchanged on error.
pub fn normal_with_seed<T, B>(
    weight: &mut Var<T, B>,
    mean: f64,
    std_dev: f64,
    seed: u64,
) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    let mean_t = finite::<T, B::Error>("mean", mean)?;
    let std_dev_t = finite::<T, B::Error>("std_dev", std_dev)?;
    if std_dev < 0.0 {
        return Err(InitializationError::NegativeStandardDeviation { value: std_dev });
    }
    normal_typed_with_seed(weight, mean_t, std_dev_t, seed)
}

/// Initialize weights with values from a normal distribution using seed 42.
///
/// # Errors
///
/// Returns the same failures as [`normal_with_seed`].
pub fn normal<T, B>(weight: &mut Var<T, B>, mean: f64, std_dev: f64) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    normal_with_seed(weight, mean, std_dev, 42)
}

fn xavier_fan<E: std::error::Error + 'static>(
    fan_in: usize,
    fan_out: usize,
) -> std::result::Result<usize, InitializationError<E>> {
    if fan_in == 0 {
        return Err(InitializationError::InvalidFan {
            parameter: "fan_in",
            value: fan_in,
        });
    }
    if fan_out == 0 {
        return Err(InitializationError::InvalidFan {
            parameter: "fan_out",
            value: fan_out,
        });
    }
    fan_in
        .checked_add(fan_out)
        .ok_or(InitializationError::FanArithmeticOverflow { fan_in, fan_out })
}

fn positive_fan<E: std::error::Error + 'static>(
    fan_in: usize,
) -> std::result::Result<usize, InitializationError<E>> {
    if fan_in == 0 {
        Err(InitializationError::InvalidFan {
            parameter: "fan_in",
            value: fan_in,
        })
    } else {
        Ok(fan_in)
    }
}

/// Apply Xavier uniform initialization using a custom seed.
///
/// # Errors
///
/// Returns [`InitializationError`] when either fan is zero, their sum
/// overflows, or uniform initialization fails.
pub fn xavier_uniform_with_seed<T, B>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    let fan = xavier_fan::<B::Error>(fan_in, fan_out)?;
    let fan = <T as Scalar>::from_usize(fan);
    let limit = (<T as Scalar>::from_f64(6.0) / fan).sqrt_val();
    uniform_typed_with_seed(weight, T::zero() - limit, limit, seed)
}

/// Apply Xavier uniform initialization using seed 42.
///
/// # Errors
///
/// Returns the same failures as [`xavier_uniform_with_seed`].
pub fn xavier_uniform<T, B>(weight: &mut Var<T, B>, fan_in: usize, fan_out: usize) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    xavier_uniform_with_seed(weight, fan_in, fan_out, 42)
}

/// Apply Xavier normal initialization using a custom seed.
///
/// # Errors
///
/// Returns [`InitializationError`] when either fan is zero, their sum
/// overflows, or normal initialization fails.
pub fn xavier_normal_with_seed<T, B>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    let fan = xavier_fan::<B::Error>(fan_in, fan_out)?;
    let fan = <T as Scalar>::from_usize(fan);
    let std_dev = (<T as Scalar>::from_f64(2.0) / fan).sqrt_val();
    normal_typed_with_seed(weight, T::zero(), std_dev, seed)
}

/// Apply Xavier normal initialization using seed 42.
///
/// # Errors
///
/// Returns the same failures as [`xavier_normal_with_seed`].
pub fn xavier_normal<T, B>(weight: &mut Var<T, B>, fan_in: usize, fan_out: usize) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    xavier_normal_with_seed(weight, fan_in, fan_out, 42)
}

/// Apply Kaiming uniform initialization using a custom seed.
///
/// # Errors
///
/// Returns [`InitializationError`] when `fan_in` is zero or uniform
/// initialization fails.
pub fn kaiming_uniform_with_seed<T, B>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    seed: u64,
) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    let fan = positive_fan::<B::Error>(fan_in)?;
    let fan = <T as Scalar>::from_usize(fan);
    let limit = (<T as Scalar>::from_f64(6.0) / fan).sqrt_val();
    uniform_typed_with_seed(weight, T::zero() - limit, limit, seed)
}

/// Apply Kaiming uniform initialization using seed 42.
///
/// # Errors
///
/// Returns the same failures as [`kaiming_uniform_with_seed`].
pub fn kaiming_uniform<T, B>(weight: &mut Var<T, B>, fan_in: usize) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    kaiming_uniform_with_seed(weight, fan_in, 42)
}

/// Apply Kaiming normal initialization using a custom seed.
///
/// # Errors
///
/// Returns [`InitializationError`] when `fan_in` is zero or normal
/// initialization fails.
pub fn kaiming_normal_with_seed<T, B>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    seed: u64,
) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    let fan = positive_fan::<B::Error>(fan_in)?;
    let fan = <T as Scalar>::from_usize(fan);
    let std_dev = (<T as Scalar>::from_f64(2.0) / fan).sqrt_val();
    normal_typed_with_seed(weight, T::zero(), std_dev, seed)
}

/// Apply Kaiming normal initialization using seed 42.
///
/// # Errors
///
/// Returns the same failures as [`kaiming_normal_with_seed`].
pub fn kaiming_normal<T, B>(weight: &mut Var<T, B>, fan_in: usize) -> InitResult<B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    kaiming_normal_with_seed(weight, fan_in, 42)
}
