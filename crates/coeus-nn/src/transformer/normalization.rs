use crate::module::{ModuleError, ModuleError::Backend};
use coeus_autograd::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

pub(super) fn layer_norm_three_dimensional<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    module: &'static str,
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    eps: f64,
) -> Result<Var<T, B>, ModuleError<B::Error>> {
    let [batch, sequence, width] = super::validation::rank_three(module, input)?;
    super::validation::affine_vector(module, "layer normalization weight", weight, width)?;
    super::validation::affine_vector(module, "layer normalization bias", bias, width)?;
    if !eps.is_finite() || eps < 0.0 {
        return Err(ModuleError::InvalidEpsilon { module });
    }

    let flattened = coeus_autograd::reshape(input, [batch * sequence, width]);
    let backend = B::default();
    let mean = coeus_ops::mean_axis(&flattened.tensor, 1, &backend)
        .map_err(|source| Backend { module, source })?;
    let centered = coeus_ops::sub(&flattened.tensor, &mean, &backend);
    let centered_squared = coeus_ops::mul(&centered, &centered, &backend);
    let mut standard_deviation = coeus_ops::mean_axis(&centered_squared, 1, &backend)
        .map_err(|source| Backend { module, source })?;
    let epsilon = Tensor::full_on([1], T::from_f64(eps), &backend);
    coeus_ops::add_assign(&mut standard_deviation, &epsilon, &backend)
        .map_err(|source| Backend { module, source })?;
    coeus_ops::sqrt_assign(&mut standard_deviation, &backend)
        .map_err(|source| Backend { module, source })?;

    let mut inverse_standard_deviation = Tensor::ones_on([batch * sequence, 1], &backend);
    coeus_ops::div_assign(
        &mut inverse_standard_deviation,
        &standard_deviation,
        &backend,
    )
    .map_err(|source| Backend { module, source })?;
    let normalized = coeus_ops::mul(&centered, &inverse_standard_deviation, &backend);
    let weight_tensor = weight.tensor.reshape([1, width]);
    let bias_tensor = bias.tensor.reshape([1, width]);
    let mut output = coeus_ops::mul(&normalized, &weight_tensor, &backend);
    coeus_ops::add_assign(&mut output, &bias_tensor, &backend)
        .map_err(|source| Backend { module, source })?;

    let normalized = coeus_autograd::layernorm(
        &flattened,
        weight,
        bias,
        output,
        normalized,
        inverse_standard_deviation,
        Tensor::full_on([1], T::from_f64(width as f64), &backend),
    );
    Ok(coeus_autograd::reshape(
        &normalized,
        [batch, sequence, width],
    ))
}
