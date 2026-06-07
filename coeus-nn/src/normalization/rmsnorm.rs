use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;
use coeus_autograd::Var;
use crate::module::Module;

/// Root Mean Square Normalization (RMSNorm) module.
///
/// Applies RMSNorm over the last dimension of a 2D tensor [N, D].
#[derive(Clone)]
pub struct RMSNorm<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Trainable scale parameter gamma: [D].
    pub weight: Var<T, B>,
    /// Small value for numerical stability.
    pub eps: f64,
    /// Cached epsilon tensor: [1].
    eps_t: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> RMSNorm<T, B> {
    /// Create a new RMSNorm layer for a given feature dimension.
    pub fn new(normalized_shape: usize, eps: f64) -> Self {
        let backend = B::default();
        let weight = Var::new(Tensor::ones_on([normalized_shape], &backend), true);
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
        Self { weight, eps, eps_t }
    }

    /// Create an RMSNorm layer from existing parameters.
    pub fn from_parts(weight: Var<T, B>, eps: f64) -> Self {
        let backend = B::default();
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
        Self { weight, eps, eps_t }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for RMSNorm<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone()]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape_cloned();
        assert_eq!(shape.len(), 2, "RMSNorm expects 2D input [batch_size, normalized_shape]");
        let _n = shape[0];
        let d = shape[1];
        let backend = B::default();

        // ── Mean square ──
        let x_sq = coeus_ops::mul(&input.tensor, &input.tensor, &backend);
        let mut rms = coeus_ops::mean_axis(&x_sq, 1, &backend); // [N, 1]

        // ── RMS ──
        coeus_ops::add_assign(&mut rms, &self.eps_t, &backend);
        coeus_ops::sqrt_assign(&mut rms, &backend);

        // ── Normalize ──
        let x_hat = coeus_ops::div(&input.tensor, &rms, &backend); // [N, D]

        // ── Scale ──
        let w_reshaped = self.weight.tensor.reshape([1, d]);
        let out_tensor = coeus_ops::mul(&x_hat, &w_reshaped, &backend);

        coeus_autograd::rmsnorm(
            input,
            &self.weight,
            out_tensor,
            x_hat,
            rms,
        )
    }
}
