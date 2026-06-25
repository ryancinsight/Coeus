use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;
use std::cell::RefCell;

/// Layer Normalization module.
///
/// Applies Layer Normalization over the last dimension of a 2D tensor `[N, D]`.
#[derive(Clone)]
pub struct LayerNorm<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Trainable scale parameter gamma: `[D]`.
    pub weight: Var<T, B>,
    /// Trainable shift parameter beta: `[D]`.
    pub bias: Var<T, B>,
    /// Small value for numerical stability.
    pub eps: f64,
    /// Cached epsilon tensor: `[1]`.
    eps_t: Tensor<T, B>,
    /// Cached dimension constant: `[1]`.
    d_const: Tensor<T, B>,
    /// Cached ones tensor of shape `[N, 1]`: (N, ones_tensor).
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

    /// Forward pass for 2-D input `[N, D]`.
    ///
    /// For inputs with rank ≥ 3 (e.g. `[batch, seq, D]`) use [`forward_nd`] instead,
    /// or call [`forward_nd`] directly — it handles all ranks ≥ 2 via reshape.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape_cloned();
        assert_eq!(
            shape.len(),
            2,
            "LayerNorm::forward expects 2D input [N, D]; for rank-N (N≥3) inputs call forward_nd"
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

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> LayerNorm<T, B> {
    /// Forward pass for any rank ≥ 2 input.
    ///
    /// Collapses all leading dimensions into a single batch dimension, applies
    /// the standard 2-D LayerNorm over the last (`D`) dimension, then restores
    /// the original shape.  All three operations use tracked `coeus_autograd::reshape`
    /// so gradients flow through the entire reshape→normalize→reshape chain.
    ///
    /// # Examples
    /// ```text
    /// // 3-D Transformer hidden states [batch, seq, d_model]:
    /// let output = layer_norm.forward_nd(&x);
    ///
    /// // 4-D activation map [batch, channels, h, w]:
    /// let output = layer_norm.forward_nd(&x);
    /// ```
    pub fn forward_nd(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape_cloned();
        let ndim = shape.len();
        assert!(
            ndim >= 2,
            "LayerNorm::forward_nd requires rank ≥ 2, got rank {ndim}"
        );
        if ndim == 2 {
            // Fast path: avoid a no-op reshape pair.
            return self.forward(input);
        }
        let d = shape[ndim - 1];
        let leading: usize = shape[..ndim - 1].iter().product();
        // Tracked flatten [... , D] → [leading, D]
        let flat = coeus_autograd::reshape(input, [leading, d]);
        // Apply 2-D LayerNorm
        let normed = self.forward(&flat);
        // Tracked unflatten back to original shape
        coeus_autograd::reshape(&normed, shape)
    }
}
