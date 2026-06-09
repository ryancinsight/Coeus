use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;
use std::cell::RefCell;

/// Layer Normalization module.
///
/// Applies Layer Normalization over the last dimension of a 2D tensor [N, D].
#[derive(Clone)]
pub struct LayerNorm<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Trainable scale parameter gamma: [D].
    pub weight: Var<T, B>,
    /// Trainable shift parameter beta: [D].
    pub bias: Var<T, B>,
    /// Small value for numerical stability.
    pub eps: f64,
    /// Cached epsilon tensor: [1].
    eps_t: Tensor<T, B>,
    /// Cached dimension constant: [1].
    d_const: Tensor<T, B>,
    /// Cached ones tensor of shape [N, 1]: (N, ones_tensor)
    ones_cache: RefCell<Option<(usize, Tensor<T, B>)>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> LayerNorm<T, B> {
    /// Create a new LayerNorm layer for a given feature dimension.
    pub fn new(normalized_shape: usize, eps: f64) -> Self {
        let backend = B::default();
        let weight = Var::new(Tensor::ones_on([normalized_shape], &backend), true);
        let bias = Var::new(Tensor::zeros_on([normalized_shape], &backend), true);
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
        let d_const = Tensor::full_on([1], T::from_f64(normalized_shape as f64), &backend);
        Self {
            weight,
            bias,
            eps,
            eps_t,
            d_const,
            ones_cache: RefCell::new(None),
        }
    }

    /// Create a LayerNorm layer from existing parameters.
    pub fn from_parts(weight: Var<T, B>, bias: Var<T, B>, eps: f64) -> Self {
        let backend = B::default();
        let normalized_shape = weight.tensor.shape()[0];
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
        let d_const = Tensor::full_on([1], T::from_f64(normalized_shape as f64), &backend);
        Self {
            weight,
            bias,
            eps,
            eps_t,
            d_const,
            ones_cache: RefCell::new(None),
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for LayerNorm<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape_cloned();
        assert_eq!(
            shape.len(),
            2,
            "LayerNorm expects 2D input [batch_size, normalized_shape]"
        );
        let _n = shape[0];
        let d = shape[1];
        let backend = B::default();

        // ── Mean over last dimension ──
        let mean_t = coeus_ops::mean_axis(&input.tensor, 1, &backend); // [N, 1]

        // ── Centered: x - mu ──
        let xmu = coeus_ops::sub(&input.tensor, &mean_t, &backend); // [N, D]

        // ── Variance ──
        let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
        let mut stdev = coeus_ops::mean_axis(&xmu_sq, 1, &backend); // [N, 1]

        // ── 1/sqrt(var + eps) ──
        coeus_ops::add_assign(&mut stdev, &self.eps_t, &backend);
        coeus_ops::sqrt_assign(&mut stdev, &backend);

        let ones = {
            let mut cache = self.ones_cache.borrow_mut();
            if let Some((cached_n, ref cached_ones)) = *cache {
                if cached_n == shape[0] {
                    cached_ones.clone()
                } else {
                    let ones = Tensor::ones_on([shape[0], 1], &backend);
                    *cache = Some((shape[0], ones.clone()));
                    ones
                }
            } else {
                let ones = Tensor::ones_on([shape[0], 1], &backend);
                *cache = Some((shape[0], ones.clone()));
                ones
            }
        };
        let mut istdev = ones;
        coeus_ops::div_assign(&mut istdev, &stdev, &backend); // [N, 1]

        // ── Normalize ──
        let x_hat = coeus_ops::mul(&xmu, &istdev, &backend); // [N, D]

        // ── Scale and bias ──
        let w_reshaped = self.weight.tensor.reshape([1, d]);
        let b_reshaped = self.bias.tensor.reshape([1, d]);
        let mut out_tensor = coeus_ops::mul(&x_hat, &w_reshaped, &backend);
        coeus_ops::add_assign(&mut out_tensor, &b_reshaped, &backend);

        coeus_autograd::layernorm(
            input,
            &self.weight,
            &self.bias,
            out_tensor,
            x_hat,
            istdev,
            self.d_const.clone(),
        )
    }
}
