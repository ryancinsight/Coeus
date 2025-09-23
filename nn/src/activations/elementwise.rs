//! Elementwise activation functions
//!
//! This module contains activation functions that operate element-wise on tensors.

use crate::Module;
use coeus_dtype::Dtype;
use coeus_tensor::{FloatDtype, Tensor};
use std::fmt;

/// ReLU (Rectified Linear Unit) activation function
///
/// Formula: `ReLU(x) = max(0, x)`
///
/// This is the most commonly used activation function in modern neural networks
/// due to its simplicity and effectiveness in combating the vanishing gradient problem.
#[derive(Debug, Clone, Copy)]
pub struct ReLU;

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ReLU {
    /// Create a new ReLU activation
    pub fn new() -> Self {
        ReLU
    }
}

impl<T: FloatDtype> Module<T> for ReLU {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Use the tensor's relu method which has proper autograd integration
        Ok(input.relu())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for ReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLU()")
    }
}

/// Leaky ReLU activation function
///
/// Formula: `LeakyReLU(x) = max(αx, x)` where α is a small positive constant
///
/// Leaky ReLU allows a small gradient when the input is negative,
/// helping to mitigate the "dying ReLU" problem.
#[derive(Debug, Clone)]
pub struct LeakyReLU {
    /// Negative slope coefficient
    negative_slope: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl LeakyReLU {
    /// Create a new LeakyReLU with default negative slope (0.01)
    pub fn new() -> Self {
        Self::new_with_slope(0.01)
    }

    /// Create a new LeakyReLU with custom negative slope
    pub fn new_with_slope(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl<T: FloatDtype> Module<T> for LeakyReLU {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let alpha = T::from_f64(self.negative_slope).unwrap();

        let data = input
            .data()
            .iter()
            .map(|&x| if x > T::zero() { x } else { alpha * x })
            .collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for LeakyReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LeakyReLU(negative_slope={})", self.negative_slope)
    }
}

/// ELU (Exponential Linear Unit) activation function
///
/// Formula: `ELU(x) = x if x > 0 else α * (exp(x) - 1)`
///
/// ELU is similar to ReLU but allows negative values, which can help with the
/// vanishing gradient problem. The default alpha value is 1.0.
#[derive(Debug, Clone, Copy)]
pub struct ELU<T: FloatDtype> {
    alpha: T,
}

impl<T: FloatDtype> Default for ELU<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> ELU<T> {
    /// Create a new ELU activation with default alpha = 1.0
    pub fn new() -> Self {
        Self {
            alpha: T::from(1.0).unwrap(),
        }
    }

    /// Create a new ELU activation with specified alpha
    pub fn with_alpha(alpha: T) -> Self {
        Self { alpha }
    }
}

impl<T: FloatDtype> Module<T> for ELU<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Ok(input.elu(self.alpha))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl<T: FloatDtype> fmt::Display for ELU<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ELU(alpha={})",
            Dtype::to_f64(&self.alpha).unwrap_or(1.0)
        )
    }
}

/// GELU (Gaussian Error Linear Unit) activation function
///
/// Formula: `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// GELU is a smooth approximation to ReLU that is commonly used in
/// transformer-based architectures like BERT and GPT.
#[derive(Debug, Clone, Copy, Default)]
pub struct GELU;

impl GELU {
    /// Create a new GELU activation
    pub fn new() -> Self {
        Self
    }
}

impl<T: FloatDtype> Module<T> for GELU {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Ok(input.gelu())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for GELU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GELU()")
    }
}

/// Hardshrink activation function
///
/// Formula: `Hardshrink(x) = x if |x| > λ else 0` where λ is the threshold
///
/// Hardshrink sets values within the range [-λ, λ] to zero, preserving larger values.
/// This can help with sparsity and regularization.
#[derive(Debug, Clone, Copy)]
pub struct Hardshrink {
    /// Threshold value λ
    lambda: f64,
}

impl Default for Hardshrink {
    fn default() -> Self {
        Self::new()
    }
}

impl Hardshrink {
    /// Create a new Hardshrink activation with default lambda = 0.5
    pub fn new() -> Self {
        Self::new_with_lambda(0.5)
    }

    /// Create a new Hardshrink activation with specified lambda
    pub fn new_with_lambda(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl<T: FloatDtype> Module<T> for Hardshrink {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let lambda = T::from_f64(self.lambda).unwrap();

        let data = input
            .data()
            .iter()
            .map(|&x| if x.abs() > lambda { x } else { T::zero() })
            .collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for Hardshrink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hardshrink(lambda={})", self.lambda)
    }
}

/// Hardtanh activation function
///
/// Formula: `Hardtanh(x) = max(min(x, max_val), min_val)`
///
/// Hardtanh is a clipped version of the identity function, constraining
/// outputs to a specified range.
#[derive(Debug, Clone, Copy)]
pub struct Hardtanh {
    /// Minimum value
    min_val: f64,
    /// Maximum value
    max_val: f64,
}

impl Default for Hardtanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Hardtanh {
    /// Create a new Hardtanh activation with default range [-1.0, 1.0]
    pub fn new() -> Self {
        Self::new_with_range(-1.0, 1.0)
    }

    /// Create a new Hardtanh activation with specified range
    pub fn new_with_range(min_val: f64, max_val: f64) -> Self {
        Self { min_val, max_val }
    }
}

impl<T: FloatDtype> Module<T> for Hardtanh {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let min_val = T::from_f64(self.min_val).unwrap();
        let max_val = T::from_f64(self.max_val).unwrap();

        let data = input
            .data()
            .iter()
            .map(|&x| x.clamp(min_val, max_val))
            .collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for Hardtanh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hardtanh(min_val={}, max_val={})", self.min_val, self.max_val)
    }
}

/// PReLU (Parametric ReLU) activation function
///
/// Formula: `PReLU(x) = max(0, x) + a * min(0, x)` where a is a learnable parameter
///
/// PReLU allows negative values with a learnable slope, providing more flexibility
/// than standard ReLU while still being computationally efficient.
#[derive(Debug, Clone)]
pub struct PReLU<T: FloatDtype> {
    /// Learnable negative slope parameter
    a: Tensor<T>,
}

impl<T: FloatDtype> Default for PReLU<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> PReLU<T> {
    /// Create a new PReLU activation with default slope 0.25
    pub fn new() -> Self {
        Self::new_with_slope(T::from(0.25).unwrap())
    }

    /// Create a new PReLU activation with specified slope
    pub fn new_with_slope(slope: T) -> Self {
        Self {
            a: Tensor::from_vec(vec![slope], vec![1]),
        }
    }
}

impl<T: FloatDtype + Clone> Module<T> for PReLU<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let a = &self.a;

        let data = input
            .data()
            .iter()
            .zip(a.data().iter().cycle())
            .map(|(&x, &slope)| if x > T::zero() { x } else { slope * x })
            .collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.a]
    }
}

impl<T: FloatDtype> fmt::Display for PReLU<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PReLU(num_parameters={})", self.a.numel())
    }
}

/// RReLU (Randomized Leaky ReLU) activation function
///
/// Formula: `RReLU(x) = max(0, x) + a * min(0, x)` where a is randomly sampled
///
/// RReLU introduces randomness during training to help with regularization,
/// using a fixed value during inference.
#[derive(Debug, Clone, Copy)]
pub struct RReLU {
    /// Lower bound for random slope
    lower: f64,
    /// Upper bound for random slope
    upper: f64,
}

impl Default for RReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl RReLU {
    /// Create a new RReLU with default range [0.125, 0.333]
    pub fn new() -> Self {
        Self::new_with_range(0.125, 0.333)
    }

    /// Create a new RReLU with specified range
    pub fn new_with_range(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }
}

impl<T: FloatDtype> Module<T> for RReLU {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // During training, use random slope; during inference, use average
        let slope = (self.lower + self.upper) / 2.0;
        let alpha = T::from_f64(slope).unwrap();

        let data = input
            .data()
            .iter()
            .map(|&x| if x > T::zero() { x } else { alpha * x })
            .collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for RReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RReLU(lower={}, upper={})", self.lower, self.upper)
    }
}

/// Tanhshrink activation function
///
/// Formula: `Tanhshrink(x) = x - tanh(x)`
///
/// Tanhshrink subtracts the tanh of the input from the input itself,
/// creating a function that grows linearly for large inputs.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanhshrink;

impl Tanhshrink {
    /// Create a new Tanhshrink activation
    pub fn new() -> Self {
        Self
    }
}

impl<T: FloatDtype> Module<T> for Tanhshrink {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let tanh_result = input.tanh();
        Ok((input - &tanh_result).unwrap())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for Tanhshrink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tanhshrink()")
    }
}

/// Threshold activation function
///
/// Formula: `Threshold(x) = x if x > threshold else value`
///
/// Threshold sets all values below a threshold to a specified value,
/// while leaving larger values unchanged.
#[derive(Debug, Clone, Copy)]
pub struct Threshold {
    /// Threshold value
    threshold: f64,
    /// Value to replace elements below threshold
    value: f64,
}

impl Default for Threshold {
    fn default() -> Self {
        Self::new()
    }
}

impl Threshold {
    /// Create a new Threshold activation with default threshold=0.0, value=0.0
    pub fn new() -> Self {
        Self::new_with_params(0.0, 0.0)
    }

    /// Create a new Threshold activation with specified parameters
    pub fn new_with_params(threshold: f64, value: f64) -> Self {
        Self { threshold, value }
    }
}

impl<T: FloatDtype> Module<T> for Threshold {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let threshold = T::from_f64(self.threshold).unwrap();
        let value = T::from_f64(self.value).unwrap();

        let data = input
            .data()
            .iter()
            .map(|&x| if x > threshold { x } else { value })
            .collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for Threshold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Threshold(threshold={}, value={})", self.threshold, self.value)
    }
}

/// CELU (Continuously Differentiable Exponential Linear Unit) activation function
///
/// Formula: `CELU(x) = max(0, x) + min(0, α * (exp(x/α) - 1))`
///
/// CELU is a continuously differentiable version of ELU with better properties
/// for optimization.
#[derive(Debug, Clone, Copy)]
pub struct CELU<T: FloatDtype> {
    alpha: T,
}

impl<T: FloatDtype> Default for CELU<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> CELU<T> {
    /// Create a new CELU activation with default alpha = 1.0
    pub fn new() -> Self {
        Self {
            alpha: T::from(1.0).unwrap(),
        }
    }

    /// Create a new CELU activation with specified alpha
    pub fn with_alpha(alpha: T) -> Self {
        Self { alpha }
    }
}

impl<T: FloatDtype> Module<T> for CELU<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let alpha = self.alpha;

        let data = input
            .data()
            .iter()
            .map(|&x| {
                if x > T::zero() {
                    x
                } else {
                    alpha * (T::exp(x / alpha) - T::one())
                }
            })
            .collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl<T: FloatDtype> fmt::Display for CELU<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CELU(alpha={})",
            Dtype::to_f64(&self.alpha).unwrap_or(1.0)
        )
    }
}

/// SELU (Scaled Exponential Linear Unit) activation function
///
/// Formula: `SELU(x) = λ * (max(0, x) + min(0, α * (exp(x) - 1)))`
/// where λ ≈ 1.0507 and α ≈ 1.6733
///
/// SELU is a self-normalizing activation function that maintains mean and
/// variance of the input, helping with training stability.
#[derive(Debug, Clone, Copy, Default)]
pub struct SELU;

impl SELU {
    /// Create a new SELU activation
    pub fn new() -> Self {
        Self
    }
}

impl<T: FloatDtype> Module<T> for SELU {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let lambda = T::from(1.050_700_987_355_480_5).unwrap(); // λ
        let alpha = T::from(1.673_263_242_354_377_2).unwrap();  // α

        let data = input
            .data()
            .iter()
            .map(|&x| {
                lambda * if x > T::zero() {
                    x
                } else {
                    alpha * (T::exp(x) - T::one())
                }
            })
            .collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for SELU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SELU()")
    }
}