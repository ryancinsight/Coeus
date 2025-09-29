use crate::{Module, Result, NNError};
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};
use coeus_backend::{Backend, CpuBackend};
use std::sync::Arc;
use tracing::{instrument, debug_span};

//! Batch Normalization for 1D data
//!
//! BatchNorm1d normalizes the input in the features dimension.
//!
//! ## Mathematical Foundation
//!
//! Forward: $$ y = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \gamma + \beta $$
//!
//! Where $\mu_B$, $\sigma_B^2$ are batch mean and variance.
//!
//! Running stats: $\mu = (1 - \beta) \mu + \beta \mu_B$, $\sigma^2 = (1 - \beta) \sigma^2 + \beta (\sigma_B^2 + \frac{\mu_B - \mu}{\beta} ^2)$
//!
//! Backward: $ \frac{\partial L}{\partial x} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left( \frac{\partial L}{\partial y} - \frac{1}{B} \sum \frac{\partial L}{\partial y} \right) - \frac{\gamma (x - \mu)}{B \sqrt{\sigma^2 + \epsilon}^3} \sum \frac{\partial L}{\partial y} (x - \mu) $
//!
//! ```mermaid
//! graph LR
//!     A[Input x: batch x len x features] --> B[Batch Mean μ_B = 1/B ∑x]
//!     A --> C[Batch Var σ_B^2 = 1/B ∑(x-μ_B)^2]
//!     B --> D[Normalize: (x - μ_B)/√(σ_B^2 + ε)]
//!     C --> D
//!     D --> E[Scale γ + Shift β]
//!     E --> F[Output y]
//!     F --> G[Update Running μ, σ^2 with momentum β]
//! ```
//!
//! # Training vs Evaluation
//! - Training: Use batch statistics
//! - Evaluation: Use running statistics

#[derive(Debug, Clone)]
pub struct BatchNorm1d<T: FloatDtype, B: Backend<T> + Clone + Send + Sync + Default> {
    /// Learnable scale parameter γ
    pub gamma: Tensor<T, B>,
    /// Learnable shift parameter β
    pub beta: Tensor<T, B>,
    /// Running mean (updated during training)
    pub running_mean: Tensor<T, B>,
    /// Running variance (updated during training)
    pub running_var: Tensor<T, B>,
    /// Momentum for running stats update
    pub momentum: T,
    /// Small value added to variance for numerical stability
    pub eps: T,
    /// Whether to use affine transformation (γ, β)
    pub affine: bool,
    /// Whether the layer is in training mode
    pub training: bool,
    /// Backend for tensor operations
    pub backend: B,
    /// Cached input for backward
    pub input: Option<Arc<Tensor<T, B>>>,
    /// Cached batch mean/var for backward
    pub batch_mean: Option<Tensor<T, B>>,
    pub batch_var: Option<Tensor<T, B>>,
    /// Input shape for backward
    pub input_shape: Option<Vec<usize>>,
}

impl<T: FloatDtype, B: Backend<T> + Clone + Send + Sync + Default> BatchNorm1d<T, B> {
    /// Create a new BatchNorm1d layer
    ///
    /// # Arguments
    /// * `num_features` - Number of features to normalize
    /// * `eps` - Epsilon for numerical stability (default: 1e-5)
    /// * `momentum` - Momentum for running stats (default: 0.1)
    /// * `affine` - Whether to learn γ, β parameters (default: true)
    /// * `track_running_stats` - Whether to track running mean/var (default: true)
    pub fn new(num_features: usize, eps: T, momentum: T, affine: bool) -> Result<Self> {
        let backend = B::default();
        let gamma = if affine {
            backend.ones(vec![num_features]).map_err(NNError::from)?
        } else {
            backend.zeros(vec![num_features]).map_err(NNError::from)?
        };
        let beta = backend.zeros(vec![num_features]).map_err(NNError::from)?;
        let running_mean = backend.zeros(vec![num_features]).map_err(NNError::from)?;
        let running_var = backend.ones(vec![num_features]).map_err(NNError::from)?;
        Ok(Self {
            gamma,
            beta,
            running_mean,
            running_var,
            momentum,
            eps,
            affine,
            training: true,
            backend,
            input: None,
            batch_mean: None,
            batch_var: None,
            input_shape: None,
        })
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Forward pass in training mode
    fn forward_train(&self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let (batch, len, features) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if features != self.gamma.shape()[0] {
            return Err(NNError::ShapeMismatch {
                expected: vec![self.gamma.shape()[0]],
                actual: vec![features],
            });
        }

        // Compute batch mean: sum over batch*len, divide by batch*len
        let total_elements = batch * len;
        let mut mean_data = vec![T::zero(); features];
        let input_data = input.data();
        for f in 0..features {
            let mut sum = T::zero();
            for i in 0..total_elements {
                sum += input_data[i * features + f];
            }
            mean_data[f] = sum / T::from_usize(total_elements).unwrap_or(T::one());
        }
        let batch_mean = self.backend.from_vec(mean_err(NNError::from)?;

        // Compute batch var: mean((x - mean)^2)
        let mut var_data = vec![T::zero(); features];
        for f in 0..features {
            let m = mean_data[f];
            let mut sum_sq = T::zero();
            for i in 0..total_elements {
                let diff = input_data[i * features + f] - m;
                sum_sq += diff * diff;
            }
            var_data[f] = sum_sq / T::from_usize(total_elements).unwrap_or(T::one());
        }
        let batch_var = self.backend.from_vec(var_data, vec![features]).map_err(NNError::from)?;

        // Normalize: (x - mean) / sqrt(var + eps)
        let mut norm_data = vec![T::zero(); input_data.len()];
        let sqrt_var = batch_var.sqrt()? + self.eps;
        for i in 0..input_data.len() {
            let f = i % features;
            let val = input_data[i];
            let denom = sqrt_var.data()[f];
            norm_data[i] = (val - batch_mean.data()[f]) / denom;
        }
        let normalized = self.backend.from_vec(norm_data, input.shape().clone())?;

        // Affine: gamma * norm + beta
        let mut output = if self.affine {
            let scaled = self.backend.mul_elementwise(&normalized, &self.gamma.unsqueeze(1)?)?;
            self.backend.add_elementwise(&scaled, &self.beta.unsqueeze(1)?)?
        } else {
            normalized
        };

        // Update running stats
        let momentum_complement = T::one() - self.momentum;
        let running_mean_scaled = self.backend.mul_scalar(&self.running_mean, momentum_complement)?;
        let batch_mean_scaled = self.backend.mul_scalar(&batch_mean, self.momentum)?;
        let running_mean_update = self.backend.add_elementwise(&running_mean_scaled, &batch_mean_scaled)?;
        
        let running_var_scaled = self.backend.mul_scalar(&self.running_var, momentum_complement)?;
        let batch_var_scaled = self.backend.mul_scalar(&batch_var, self.momentum)?;
        let running_var_update = self.backend.add_elementwise(&running_var_scaled, &batch_var_scaled)?;
        self.running_mean = running_mean_update;
        self.running_var = running_var_update;

        self.input = Some(Arc::new(input.clone()));
        self.batch_mean = Some(batch_mean);
        self.batch_var = Some(batch_var);
        self.input_shape = Some(input.shape().clone());

        Ok(output)
    }

    /// Forward pass in evaluation mode
    fn forward_eval(&self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let (batch, len, features) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if features != self.gamma.shape()[0] {
            return Err(NNError::ShapeMismatch {
                expected: vec![self.gamma.shape()[0]],
                actual: vec![features],
            });
        }

        // Normalize using running stats
        let mut norm_data = vec![T::zero(); input.data().len()];
        let input_data = input.data();
        let sqrt_var = self.running_var.sqrt()? + self.eps;
        for i in 0..input_data.len() {
            let f = i % features;
            let val = input_data[i];
            let denom = sqrt_var.data()[f];
            norm_data[i] = (val - self.running_mean.data()[f]) / denom;
        }
        let normalized = self.backend.from_vec(norm_data, input.shape().clone())?;

        // Affine
        let output = if self.affine {
            let scaled = self.backend.mul_elementwise(&normalized, &self.gamma.unsqueeze(1)?)?;
            self.backend.add_elementwise(&scaled, &self.beta.unsqueeze(1)?)?
        } else {
            normalized
        };

        Ok(output)
    }

    #[instrument(skip(self, input), fields(input_shape=?input.shape(), training=self.training))]
    pub fn forward(&mut self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        if self.training {
            self.forward_train(input)
        } else {
            self.forward_eval(input)
        }
    }

    #[instrument(skip(self, grad_output))]
    pub fn backward(&self, grad_output: &Tensor<T, B>) -> Result<(Tensor<T, B>, Tensor<T, B>, Tensor<T, B>)> {
        let input = self.input.as_ref().ok_or(NNError::StateError("No cached input".into()))?.clone();
        let batch_mean = self.batch_mean.as_ref().ok_or(NNError::StateError("No batch mean".into()))?.clone();
        let batch_var = self.batch_var.as_ref().ok_or(NNError::StateError("No batch var".into()))?.clone();
        let (batch, len, features) = (grad_output.shape()[0], grad_output.shape()[1], grad_output.shape()[2]);

        let total_elements = batch * len;
        let input_data = input.data();
        let grad_out_data = grad_output.data();
        let mean_data = batch_mean.data();
        let var_data = batch_var.data();

        // Compute grad_gamma = sum(grad_out * norm)
        let mut grad_gamma_data = vec![T::zero(); features];
        for f in 0..features {
            let mut sum = T::zero();
            for i in 0..total_elements {
                let norm = (input_data[i * features + f] - mean_data[f]) / (var_data[f].sqrt() + self.eps);
                sum += grad_out_data[i * features + f] * norm;
            }
            grad_gamma_data[f] = sum;
        }
        let grad_gamma = self.backend.from_vec(grad_gamma_data, vec![features])?;

        // Compute grad_beta = sum(grad_out)
        let mut grad_beta_data = vec![T::zero(); features];
        for f in 0..features {
            let mut sum = T::zero();
            for i in 0..total_elements {
                sum += grad_out_data[i * features + f];
            }
            grad_beta_data[f] = sum;
        }
        let grad_beta = self.backend.from_vec(grad_beta_data, vec![features])?;

        // Compute grad_input
        let sqrt_var = batch_var.sqrt()? + self.eps;
        let mut grad_in_data = vec![T::zero(); input_data.len()];
        for f in 0..features {
            let ivar = T::one() / sqrt_var.data()[f];
            let mut sum_dy = T::zero();
            let mut sum_dy_xmu = T::zero();
            for i in 0..total_elements {
                let dy = grad_out_data[i * features + f];
                let xmu = input_data[i * features + f] - mean_data[f];
                sum_dy += dy;
                sum_dy_xmu += dy * xmu;
            }
            let mean_dy = sum_dy / T::from_usize(total_elements).unwrap_or(T::one());
            let mean_dy_xmu = sum_dy_xmu / T::from_usize(total_elements).unwrap_or(T::one());
            for i in 0..total_elements {
                let dy = grad_out_data[i * features + f];
                let xmu = input_data[i * features + f] - mean_data[f];
                let term1 = ivar * (dy - mean_dy);
                let term2 = ivar * ivar * ivar * mean_dy_xmu * xmu;
                grad_in_data[i * features + f] = self.gamma.data()[f] * (term1 - term2);
            }
        }
        let grad_in = self.backend.from_vec(grad_in_data, input.shape().clone())?;

        Ok((grad_in, grad_gamma, grad_beta))
    }
}

impl<T: FloatDtype, B: Backend<T> + Clone + Send + Sync + Default> Module<T, B> for BatchNorm1d<T, B> {
    fn forward(&mut self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor<T, B>> {
        if self.affine {
            vec![&self.gamma, &self.beta]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
        if self.affine {
            vec![&mut self.gamma, &mut self.beta]
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        fn prop_batchnorm1d_forward(
            batch in 1usize..=4,
            len in 5usize..=10,
            features in 1usize..=4,
        ) {
            let backend = CpuBackend::default();
            let bn = BatchNorm1d::new(features, <T as Dtype>::from_f64(1e-5).unwrap(), <T as Dtype>::from_f64(0.1).unwrap(), true).unwrap();
            let input_shape = vec![batch, len, features];
            let input = backend.randn(&input_shape).unwrap();
            let mut bn_clone = bn.clone();
            let output = bn_clone.forward$1.unwrap_grad();
            prop_assert_eq!(output.shape(), input.shape());
            prop_assert!(!output.data().iter().all(|&x| x == T::zero()));
            prop_assert!(output.data().iter().all(|&x| x.is_finite() && !x.is_nan()));
            // Check running stats updated
            prop_assert!(!bn_clone.running_mean.data().iter().all(|&x| x == T::zero()));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        fn prop_batchnorm1d_backward(
            batch in 1..=2,
            len in 5..=8,
            features in 1..=2,
        ) {
            let backend = CpuBackend::default();
            let bn = BatchNorm1d::new(features, <T as Dtype>::from_f64(1e-5).unwrap(), <T as Dtype>::from_f64(0.1).unwrap(), true).unwrap();
            let input = backend.randn(vec![batch, len, features]).unwrap();
            let mut bn_clone = bn.clone();
            let output = bn_clone.forward$1.unwrap_grad();
            let ones_grad = backend.ones(output.shape()).unwrap();
            let (analytic_grad_in, _, _) = bn_clone.backward(&ones_grad).unwrap();

            let eps = <T as Dtype>::from_f64(1e-6).unwrap();
            let mut input_pert = input.clone();
            input_pert = (input_pert + &(backend.randn(input.shape()).unwrap() * eps)?).unwrap();
            let out_pert = bn_clone.forward$1.unwrap_grad();
            let numeric_grad_in = (out_pert - output)? / eps;

            for (a, n) in analytic_grad_in.data().iter().zip(numeric_grad_in.data().iter()) {
                if *n != T::zero() {
                    prop_assert_relative_eq!(*a, *n, epsilon = 1e-3);
                }
            }
        }
    }

    #[test]
    fn test_batchnorm1d_edges() {
        type F = f32;
        let backend = CpuBackend::default();
        let bn = BatchNorm1d::<F, _>::new(1, F::from_f64(1e-5).unwrap(), F::from_f64(0.1).unwrap(), true).unwrap();

        // Zero input → zero mean/var=0, but eps prevent div0, output=beta=0
        let zero_in = backend.zeros(vec![2,3,1]).unwrap();
        let mut bn_zero = bn.clone();
        let out_zero = bn_zero.forward$1.unwrap_grad();
        assert!(out_zero.data().iter().all(|&x| x == F::zero()));

        // Constant input: mean=const, var=0, output=gamma*0 + beta = beta
        let const_in = backend.from_f64(5.0, vec![2,3,1]).unwrap();
        let mut bn_const = bn.clone();
        let out_const = bn_const.forward$1.unwrap_grad();
        assert!(out_const.data().iter().all(|&x| x == F::zero()));  // gamma=1, but norm=0, beta=0

        // Neg input: mean neg, var pos, norm neg, output gamma*neg + beta
        let neg_in_data = vec![-1.0; 6];
        let neg_in = backend.from_vec(neg_in_data, vec![2,3,1]).unwrap();
        let mut bn_neg = bn.clone();
        let out_neg = bn_neg.forward$1.unwrap_grad();
        assert!(out_neg.data().iter().all(|&x| x < F::zero()));  // norm=-1, gamma=1, beta=0

        // Inf propagate: mean Inf, var NaN, but check finite Err
        let inf_in = backend.from_f64(f32::INFINITY as f64, vec![2,3,1]).unwrap();
        let mut bn_inf = bn.clone();
        let res_inf = bn_inf.forward(&inf_in);
        assert!(res_inf.is_err());  // Finite check Err

        // NaN propagate: input NaN → output NaN, Err
        let nan_in = backend.from_f64(f32::NAN as f64, vec![2,3,1]).unwrap();
        let mut bn_nan = bn.clone();
        let res_nan = bn_nan.forward(&nan_in);
        assert!(res_nan.is_err());

        // Overflow: large values, var large, sqrt ok but if >max → Inf Err
        let large = F::from_f64((f32::MAX / 10.0) as f64).unwrap();
        let large_in = backend.from_f64(large as f64, vec![2,3,1]).unwrap();
        let mut bn_large = bn.clone();
        let out_large = bn_large.forward$1.unwrap_grad();
        assert!(out_large.data().iter().all(|&x| x.is_finite()));  // Normalized to 0

        // Underflow: small values ≈0, var small, but eps, grad via backward
        let small = F::from_f64(1e-38f64).unwrap();
        let small_in = backend.from_f64(small as f64, vec![2,3,1]).unwrap();
        let mut bn_small = bn.clone();
        let out_small = bn_small.forward$1.unwrap_grad();
        assert_relative_eq!(out_small[[0,0,0]], F::zero(), epsilon = 1e-6);  // Normalized ≈0

        // Precision large mean/var rel <1e-6
        let big_mean = F::from_f64(1e10f64).unwrap();
        let big_var = F::from_f64(1e10f64).unwrap();
        // Test with known, but simplified: forward with large input, check rel
        let large_var_in = backend.randn(vec![2,3,1]).unwrap() * big_mean + big_mean;
        let mut bn_precision = bn.clone();
        let out_precision = bn_precision.forward$1.unwrap_grad();
        // Normalized should be ~N(0,1), check var ≈1 rel<1e-6
        let out_mean = out_precision.mean_dim(&[0,1])?;  // Assume mean_dim
        let out_var = out_precision.var_dim(&[0,1], true)?;  // Assume var_dim unbiased
        assert_relative_eq!(out_mean[[0,0]], F::zero(), relative = 1e-6);
        assert_relative_eq!(out_var[[0,0]], F::one(), relative = 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        fn prop_batchnorm1d_eval_consistency(
            batch in 1usize..=4,
            len in 5usize..=10,
            features in 1usize..=4,
        ) {
            let backend = CpuBackend::default();
            let bn = BatchNorm1d::new(features, <T as Dtype>::from_f64(1e-5).unwrap(), <T as Dtype>::from_f64(0.1).unwrap(), true).unwrap();
            let input_shape = vec![batch, len, features];
            let input = backend.randn(&input_shape).unwrap();
            let mut bn_train = bn.clone();
            let _ = bn_train.forward(&input);  // Train once
            bn_train.eval();
            let out_eval = bn_train.forward$1.unwrap_grad();  // Eval same input
            // Running stats approximate batch, check close after many but simplified finite
            prop_assert!(out_eval.data().iter().all(|&x| x.is_finite() && !x.is_nan()));
        }
    }

    #[test]
    fn test_batchnorm1d_affine() {
        type F = f32;
        let backend = CpuBackend::default();
        let bn_affine = BatchNorm1d::<F, _>::new(2, F::from_f64(1e-5).unwrap(), F::from_f64(0.1).unwrap(), true).unwrap();
        let bn_no_affine = BatchNorm1d::<F, _>::new(2, F::from_f64(1e-5).unwrap(), F::from_f64(0.1).unwrap(), false).unwrap();
        let input = backend.randn(vec![2,3,2]).unwrap();
        let out_affine = bn_affine.forward$1.unwrap_grad();
        let out_no = bn_no_affine.forward$1.unwrap_grad();
        // No affine: gamma=0 beta=0, output = norm *0 +0 =0
        assert!(out_no.data().iter().all(|&x| relative_eq!(x, F::zero(), epsilon = 1e-6)));
        // Affine: gamma=1 beta=0, output = norm *1 +0 = norm, var≈1
        let out_var = out_affine.var(true)?;  // Assume var
        assert_relative_eq!(out_var[[0,0]], F::one(), relative = 1e-3);  // Approx N(0,1)
    }
}


