//! Distribution-based loss functions
//!
//! Loss functions for comparing probability distributions and
//! measuring divergences between distributions.

use super::{utils, Module, Reduction};
use coeus_tensor::{FloatDtype, Tensor};

/// Kullback-Leibler Divergence Loss
///
/// Computes KL divergence between two probability distributions:
/// `KL(p||q) = Σ pᵢ * log(pᵢ / qᵢ)`
///
/// This measures how much one distribution differs from another.
/// Commonly used in variational autoencoders and policy gradient methods.
///
/// # Mathematical Properties
/// - Non-negative: KL(p||q) >= 0
/// - Non-symmetric: KL(p||q) ≠ KL(q||p) in general
/// - KL(p||q) = 0 if and only if p = q almost everywhere
/// - Gradient w.r.t. q: `∂KL/∂qᵢ = -pᵢ/qᵢ`
///
/// # Example
/// ```rust
/// use coeus_nn::KLDivLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = KLDivLoss::new();
/// let p = Tensor::from_vec(vec![0.3, 0.7], vec![2]); // Target distribution
/// let q = Tensor::from_vec(vec![0.5, 0.5], vec![2]); // Predicted distribution
///
/// let loss = loss_fn.forward(&p, &q).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct KLDivLoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl KLDivLoss {
    /// Create a new KL divergence loss with mean reduction
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create a new KL divergence loss with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute KL divergence between two probability distributions
    ///
    /// # Arguments
    /// * `p` - Target probability distribution
    /// * `q` - Predicted probability distribution
    ///
    /// # Returns
    /// KL divergence loss: KL(p||q) = Σ pᵢ * log(pᵢ / qᵢ)
    pub fn forward<T: FloatDtype>(&self, p: &Tensor<T>, q: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if p.shape() != q.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: q.shape().to_vec(),
                actual: p.shape().to_vec(),
            });
        }

        let p_data = p.data();
        let q_data = q.data();

        let mut kl_values = Vec::with_capacity(p.numel());

        for (&p_val, &q_val) in p_data.iter().zip(q_data.iter()) {
            // Clamp values to avoid log(0) and division by zero
            let p_clamped = utils::clamp_for_log(p_val);
            let q_clamped = utils::clamp_for_log(q_val);

            // KL(p||q) = p * log(p / q) = p * (log(p) - log(q))
            let ratio = p_clamped / q_clamped;
            let kl_term = p_clamped * ratio.ln();
            kl_values.push(kl_term);
        }

        let kl_tensor = Tensor::from_vec(kl_values, p.shape().to_vec());
        utils::apply_reduction(&kl_tensor, self.reduction)
    }

    /// Compute gradients of KL divergence with respect to q
    ///
    /// # Mathematical Derivation
    /// For KL(p||q) = Σ pᵢ * log(pᵢ / qᵢ):
    /// `∂KL/∂qᵢ = -pᵢ / qᵢ`
    pub fn backward<T: FloatDtype>(
        &self,
        p: &Tensor<T>,
        q: &Tensor<T>,
    ) -> crate::Result<Tensor<T>> {
        if p.shape() != q.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: q.shape().to_vec(),
                actual: p.shape().to_vec(),
            });
        }

        let p_data = p.data();
        let q_data = q.data();

        let mut grad_values = Vec::with_capacity(q.numel());

        for (&p_val, &q_val) in p_data.iter().zip(q_data.iter()) {
            // Clamp q to avoid division by zero
            let q_clamped = utils::clamp_for_log(q_val);

            // Gradient: ∂KL/∂qᵢ = -pᵢ / qᵢ
            let grad = -p_val / q_clamped;
            grad_values.push(grad);
        }

        let grad_tensor = Tensor::from_vec(grad_values, q.shape().to_vec());

        // Apply reduction scaling
        let scale = match self.reduction {
            Reduction::None => T::one(),
            Reduction::Sum => T::one(),
            Reduction::Mean => {
                let n = T::from(q.numel()).unwrap();
                T::one() / n
            }
        };

        Ok(grad_tensor.map(|x| *x * scale))
    }
}

impl<T: FloatDtype> Module<T> for KLDivLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "KLDivLoss should be used via forward() method with two inputs".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

/// Jensen-Shannon Divergence Loss
///
/// Computes Jensen-Shannon divergence between two probability distributions:
/// `JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m)` where `m = 0.5 * (p + q)`
///
/// This is a symmetric version of KL divergence, commonly used in GANs.
///
/// # Mathematical Properties
/// - Symmetric: JS(p||q) = JS(q||p)
/// - Bounded: 0 ≤ JS(p||q) ≤ log(2)
/// - Smooth and differentiable
/// - More stable than KL divergence for optimization
///
/// # Example
/// ```rust
/// use coeus_nn::JSLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = JSLoss::new();
/// let p = Tensor::from_vec(vec![0.3, 0.7], vec![2]); // First distribution
/// let q = Tensor::from_vec(vec![0.5, 0.5], vec![2]); // Second distribution
///
/// let loss = loss_fn.forward(&p, &q).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct JSLoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl JSLoss {
    /// Create a new Jensen-Shannon divergence loss with mean reduction
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create a new Jensen-Shannon divergence loss with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute Jensen-Shannon divergence between two probability distributions
    ///
    /// # Arguments
    /// * `p` - First probability distribution
    /// * `q` - Second probability distribution
    ///
    /// # Returns
    /// Jensen-Shannon divergence: JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m)
    pub fn forward<T: FloatDtype>(&self, p: &Tensor<T>, q: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if p.shape() != q.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: q.shape().to_vec(),
                actual: p.shape().to_vec(),
            });
        }

        let p_data = p.data();
        let q_data = q.data();

        let mut js_values = Vec::with_capacity(p.numel());
        let half = T::from(0.5).unwrap();

        for (&p_val, &q_val) in p_data.iter().zip(q_data.iter()) {
            // Compute mixture distribution: m = 0.5 * (p + q)
            let m_val = half * (p_val + q_val);

            // Clamp values for numerical stability
            let p_clamped = utils::clamp_for_log(p_val);
            let q_clamped = utils::clamp_for_log(q_val);
            let m_clamped = utils::clamp_for_log(m_val);

            // JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m)
            let kl_p_m = p_clamped * (p_clamped / m_clamped).ln();
            let kl_q_m = q_clamped * (q_clamped / m_clamped).ln();
            let js_term = half * (kl_p_m + kl_q_m);

            js_values.push(js_term);
        }

        let js_tensor = Tensor::from_vec(js_values, p.shape().to_vec());
        utils::apply_reduction(&js_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for JSLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "JSLoss should be used via forward() method with two inputs".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kl_div_loss_basic() {
        let loss_fn = KLDivLoss::new();
        let p = Tensor::from_vec(vec![0.3, 0.7], vec![2]); // Target distribution
        let q = Tensor::from_vec(vec![0.5, 0.5], vec![2]); // Predicted distribution

        let loss = loss_fn.forward(&p, &q).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_kl_div_loss_identical_distributions() {
        let loss_fn = KLDivLoss::new();
        let p = Tensor::from_vec(vec![0.3, 0.7], vec![2]);
        let q = Tensor::from_vec(vec![0.3, 0.7], vec![2]);

        let loss = loss_fn.forward(&p, &q).unwrap();
        // Should be very close to zero for identical distributions
        assert!(loss.item().unwrap() < 1e-10);
    }

    #[test]
    fn test_kl_div_loss_backward() {
        let loss_fn = KLDivLoss::new();
        let p = Tensor::from_vec(vec![0.3, 0.7], vec![2]);
        let q = Tensor::from_vec(vec![0.5, 0.5], vec![2]);

        let grad = loss_fn.backward(&p, &q).unwrap();
        assert_eq!(grad.shape(), q.shape());

        // Gradient should be -p/q
        let expected_grad_0 = -0.3 / 0.5;
        let expected_grad_1 = -0.7 / 0.5;

        assert_relative_eq!(grad.data()[0], expected_grad_0 / 2.0, epsilon = 1e-6); // Divided by 2 for mean reduction
        assert_relative_eq!(grad.data()[1], expected_grad_1 / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_js_loss_basic() {
        let loss_fn = JSLoss::new();
        let p = Tensor::from_vec(vec![0.3, 0.7], vec![2]);
        let q = Tensor::from_vec(vec![0.5, 0.5], vec![2]);

        let loss = loss_fn.forward(&p, &q).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_js_loss_symmetry() {
        let loss_fn = JSLoss::new();
        let p = Tensor::from_vec(vec![0.3, 0.7], vec![2]);
        let q = Tensor::from_vec(vec![0.5, 0.5], vec![2]);

        let loss_pq = loss_fn.forward(&p, &q).unwrap();
        let loss_qp = loss_fn.forward(&q, &p).unwrap();

        // JS divergence should be symmetric
        assert_relative_eq!(
            loss_pq.item().unwrap(),
            loss_qp.item().unwrap(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_js_loss_identical_distributions() {
        let loss_fn = JSLoss::new();
        let p = Tensor::from_vec(vec![0.3, 0.7], vec![2]);
        let q = Tensor::from_vec(vec![0.3, 0.7], vec![2]);

        let loss = loss_fn.forward(&p, &q).unwrap();
        // Should be very close to zero for identical distributions
        assert!(loss.item().unwrap() < 1e-10);
    }

    #[test]
    fn test_js_loss_bounded() {
        let loss_fn = JSLoss::new();
        // Maximally different distributions
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let q = Tensor::from_vec(vec![0.0, 1.0], vec![2]);

        let loss = loss_fn.forward(&p, &q).unwrap();
        let log2 = 2.0f32.ln();

        // JS divergence should be bounded by log(2)
        assert!(loss.item().unwrap() <= log2 + 1e-6);
    }

    #[test]
    fn test_distribution_losses_reductions() {
        let p = Tensor::from_vec(vec![0.3, 0.7, 0.4, 0.6], vec![2, 2]);
        let q = Tensor::from_vec(vec![0.5, 0.5, 0.3, 0.7], vec![2, 2]);

        // Test different reductions for KLDivLoss
        let loss_none = KLDivLoss::with_reduction(Reduction::None);
        let loss_sum = KLDivLoss::with_reduction(Reduction::Sum);
        let loss_mean = KLDivLoss::with_reduction(Reduction::Mean);

        let result_none = loss_none.forward(&p, &q).unwrap();
        let result_sum = loss_sum.forward(&p, &q).unwrap();
        let result_mean = loss_mean.forward(&p, &q).unwrap();

        // None should return per-element losses
        assert_eq!(result_none.shape(), &[2, 2]);

        // Sum should be sum of all elements
        let expected_sum: f32 = result_none.data().iter().sum();
        assert_relative_eq!(result_sum.item().unwrap(), expected_sum, epsilon = 1e-6);

        // Mean should be average of all elements
        let expected_mean = expected_sum / 4.0;
        assert_relative_eq!(result_mean.item().unwrap(), expected_mean, epsilon = 1e-6);
    }
}
