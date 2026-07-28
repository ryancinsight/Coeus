// ── Weight initialization ──

use coeus_autograd::Var;
use coeus_core::{BackendError, Float, Scalar};
use coeus_tensor::Tensor;

/// Initialize weights with values from a uniform distribution U(a, b).
pub fn uniform_with_seed<
    T: Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    weight: &mut Var<T, B>,
    a: f64,
    b: f64,
    seed: u64,
) -> Result<(), B::Error> {
    let shape = weight.tensor.shape_cloned();
    let values = coeus_leto::uniform_values(
        &shape,
        <T as Scalar>::from_f64(a),
        <T as Scalar>::from_f64(b),
        seed,
    )
    .map_err(|error| B::Error::from(BackendError::Storage {
        operation: "uniform initialization",
        reason: error.to_string(),
    }))?;
    weight.tensor = Tensor::from_slice_on(shape, &values, &B::default())?;
    Ok(())
}

/// Initialize weights with values from a uniform distribution U(a, b) using default seed.
pub fn uniform<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    a: f64,
    b: f64,
) -> Result<(), B::Error> {
    uniform_with_seed(weight, a, b, 42)
}

/// Initialize weights with values from a normal distribution N(mean, std_dev).
pub fn normal_with_seed<
    T: Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    weight: &mut Var<T, B>,
    mean: f64,
    std_dev: f64,
    seed: u64,
) -> Result<(), B::Error> {
    let shape = weight.tensor.shape_cloned();
    let values = coeus_leto::normal_values(
        &shape,
        <T as Scalar>::from_f64(mean),
        <T as Scalar>::from_f64(std_dev),
        seed,
    )
    .map_err(|error| B::Error::from(BackendError::Storage {
        operation: "normal initialization",
        reason: error.to_string(),
    }))?;
    weight.tensor = Tensor::from_slice_on(shape, &values, &B::default())?;
    Ok(())
}

/// Initialize weights with values from a normal distribution N(mean, std_dev) using default seed.
pub fn normal<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    mean: f64,
    std_dev: f64,
) -> Result<(), B::Error> {
    normal_with_seed(weight, mean, std_dev, 42)
}

/// Initialize weights with a constant value.
pub fn constant<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>, val: f64) -> Result<(), B::Error> {
    let shape = weight.tensor.shape_cloned();
    weight.tensor = Tensor::full_on(shape, T::from_f64(val), &B::default())?;
    Ok(())
}

/// Initialize weights with zeros.
pub fn zeros<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>) -> Result<(), B::Error> {
    let shape = weight.tensor.shape_cloned();
    weight.tensor = Tensor::zeros_on(shape, &B::default())?;
    Ok(())
}

/// Initialize weights with ones.
pub fn ones<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>) -> Result<(), B::Error> {
    let shape = weight.tensor.shape_cloned();
    weight.tensor = Tensor::ones_on(shape, &B::default())?;
    Ok(())
}

/// Xavier (Glorot) uniform initialization with custom seed.
pub fn xavier_uniform_with_seed<
    T: Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) -> Result<(), B::Error> {
    let limit = (6.0f64 / (fan_in + fan_out) as f64).sqrt();
    uniform_with_seed(weight, -limit, limit, seed)
}

/// Xavier (Glorot) uniform initialization.
pub fn xavier_uniform<
    T: Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    fan_out: usize,
) -> Result<(), B::Error> {
    xavier_uniform_with_seed(weight, fan_in, fan_out, 42)
}

/// Xavier (Glorot) normal initialization with custom seed.
pub fn xavier_normal_with_seed<
    T: Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) -> Result<(), B::Error> {
    let std_dev = (2.0f64 / (fan_in + fan_out) as f64).sqrt();
    normal_with_seed(weight, 0.0, std_dev, seed)
}

/// Xavier (Glorot) normal initialization.
pub fn xavier_normal<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    fan_out: usize,
) -> Result<(), B::Error> {
    xavier_normal_with_seed(weight, fan_in, fan_out, 42)
}

/// Kaiming (He) uniform initialization with custom seed.
pub fn kaiming_uniform_with_seed<
    T: Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    seed: u64,
) -> Result<(), B::Error> {
    let limit = (6.0f64 / fan_in as f64).sqrt();
    uniform_with_seed(weight, -limit, limit, seed)
}

/// Kaiming (He) uniform initialization.
pub fn kaiming_uniform<
    T: Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    weight: &mut Var<T, B>,
    fan_in: usize,
) -> Result<(), B::Error> {
    kaiming_uniform_with_seed(weight, fan_in, 42)
}

/// Kaiming (He) normal initialization with custom seed.
pub fn kaiming_normal_with_seed<
    T: Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    seed: u64,
) -> Result<(), B::Error> {
    let std_dev = (2.0f64 / fan_in as f64).sqrt();
    normal_with_seed(weight, 0.0, std_dev, seed)
}

/// Kaiming (He) normal initialization.
pub fn kaiming_normal<
    T: Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    weight: &mut Var<T, B>,
    fan_in: usize,
) -> Result<(), B::Error> {
    kaiming_normal_with_seed(weight, fan_in, 42)
}
