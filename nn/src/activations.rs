//! Activation functions for neural networks
//!
//! This module provides common activation functions used in neural networks.
//! All activations implement the `Module` trait for seamless integration.
//!
//! ## Available Activations
//!
//! - **ReLU**: Rectified Linear Unit, `max(0, x)`
//! - **Sigmoid**: Logistic function, `1 / (1 + exp(-x))`
//! - **Tanh**: Hyperbolic tangent, `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
//! - **Softmax**: Normalized exponential function
//! - **LeakyReLU**: Leaky version of ReLU, `max(αx, x)`
//!
//! ## Mathematical Properties
//!
//! ### ReLU
//! ```math
//! ReLU(x) = max(0, x)
//!
//! ∂ReLU/∂x = {
//!     1  if x > 0
//!     0  if x ≤ 0
//! }
//! ```
//!
//! ### Sigmoid
//! ```math
//! σ(x) = 1 / (1 + exp(-x))
//!
//! ∂σ/∂x = σ(x) * (1 - σ(x))
//! ```
//!
//! ## References
//!
//! - [Deep Learning Book - Activation Functions](https://www.deeplearningbook.org/contents/mlp.html)
//! - [Glorot & Bengio, 2010 - Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
//! - [Hendrycks & Gimpel, 2016 - Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)

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

/// Sigmoid activation function
///
/// Formula: `σ(x) = 1 / (1 + exp(-x))`
///
/// The sigmoid function squashes the input to the range (0, 1).
/// It's commonly used in binary classification problems.
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Sigmoid {
    /// Create a new Sigmoid activation
    pub fn new() -> Self {
        Sigmoid
    }
}

impl<T: FloatDtype> Module<T> for Sigmoid {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Sigmoid: 1 / (1 + exp(-x))
        let data = input
            .data()
            .iter()
            .map(|&x| T::one() / (T::one() + (-x).exp()))
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

impl fmt::Display for Sigmoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sigmoid()")
    }
}

/// Tanh (Hyperbolic Tangent) activation function
///
/// Formula: `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
///
/// Tanh squashes the input to the range (-1, 1) and is zero-centered,
/// making it preferable to sigmoid in many cases.
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Tanh {
    /// Create a new Tanh activation
    pub fn new() -> Self {
        Tanh
    }
}

impl<T: FloatDtype> Module<T> for Tanh {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        let data = input.data().iter().map(|&x| x.tanh()).collect();

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

impl fmt::Display for Tanh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tanh()")
    }
}

/// Softmax activation function
///
/// Formula: `Softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)`
///
/// Softmax converts a vector of real numbers into a probability distribution.
/// It's commonly used in the output layer of classification networks.
#[derive(Debug, Clone, Copy)]
pub struct Softmax<T: FloatDtype> {
    /// Dimension along which to apply softmax
    dim: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FloatDtype> Default for Softmax<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> Softmax<T> {
    /// Create a new Softmax activation with default dimension (-1, last dimension)
    pub fn new() -> Self {
        Self {
            dim: usize::MAX,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new Softmax activation along a specific dimension
    pub fn new_with_dim(dim: usize) -> Self {
        Self {
            dim,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: FloatDtype + Clone> Module<T> for Softmax<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let dim = if self.dim == usize::MAX {
            input.ndim() - 1
        } else {
            self.dim
        };

        // For simplicity, handle 1D and 2D cases
        if input.ndim() == 1 {
            Ok(self.softmax_1d(input))
        } else if input.ndim() == 2 && dim == 1 {
            Ok(self.softmax_2d(input))
        } else {
            // For higher dimensions, apply softmax along the specified dimension
            Ok(self.softmax_nd(input, dim))
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl<T: FloatDtype + Clone> Softmax<T> {
    fn softmax_1d(&self, input: &Tensor<T>) -> Tensor<T> {
        // exp(x) / sum(exp(x))
        let exp_data: Vec<T> = input.data().iter().map(|&x| x.exp()).collect();
        let sum_exp: T = exp_data.iter().fold(T::zero(), |acc, &x| acc + x);

        let softmax_data: Vec<T> = exp_data.iter().map(|&x| x / sum_exp).collect();

        let mut result = Tensor::from_vec(softmax_data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    fn softmax_2d(&self, input: &Tensor<T>) -> Tensor<T> {
        let shape = input.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];

        let mut result_data = vec![T::zero(); input.numel()];

        for b in 0..batch_size {
            // Extract the row for this batch
            let start_idx = b * num_classes;
            let end_idx = (b + 1) * num_classes;
            let row_data = &input.data()[start_idx..end_idx];

            // Compute exp and sum
            let exp_row: Vec<T> = row_data.iter().map(|&x| x.exp()).collect();
            let sum_exp: T = exp_row.iter().fold(T::zero(), |acc, &x| acc + x);

            // Compute softmax
            for i in 0..num_classes {
                result_data[start_idx + i] = exp_row[i] / sum_exp;
            }
        }

        let mut result = Tensor::from_vec(result_data, shape.to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    fn softmax_nd(&self, input: &Tensor<T>, dim: usize) -> Tensor<T> {
        // For general N-dimensional softmax, we need to handle broadcasting
        // This is a simplified implementation for common cases
        let shape = input.shape();
        let result_shape = shape.to_vec();

        // Compute the size along the softmax dimension
        let dim_size = shape[dim];

        // Compute total number of softmax operations needed
        let outer_size: usize = shape.iter().take(dim).product();
        let inner_size: usize = shape.iter().skip(dim + 1).product();

        let mut result_data = vec![T::zero(); input.numel()];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // Extract the vector along the softmax dimension
                let mut vector = vec![T::zero(); dim_size];
                #[allow(clippy::needless_range_loop)]
                for i in 0..dim_size {
                    let mut indices = vec![0; shape.len()];
                    // Set outer dimensions
                    let mut remaining = outer;
                    for d in (0..dim).rev() {
                        indices[d] = remaining % shape[d];
                        remaining /= shape[d];
                    }
                    // Set inner dimensions
                    remaining = inner;
                    for d in (dim + 1..shape.len()).rev() {
                        indices[d] = remaining % shape[d];
                        remaining /= shape[d];
                    }
                    indices[dim] = i;

                    // Convert to flat index
                    let mut flat_idx = 0;
                    let mut stride = 1;
                    for (d, &idx) in indices.iter().enumerate().rev() {
                        flat_idx += idx * stride;
                        stride *= shape[shape.len() - 1 - d];
                    }

                    vector[i] = input.data()[flat_idx];
                }

                // Apply softmax to this vector
                let exp_vector: Vec<T> = vector.iter().map(|&x| x.exp()).collect();
                let sum_exp: T = exp_vector.iter().fold(T::zero(), |acc, &x| acc + x);
                let softmax_vector: Vec<T> = exp_vector.iter().map(|&x| x / sum_exp).collect();

                // Store results
                #[allow(clippy::needless_range_loop)]
                for i in 0..dim_size {
                    let mut indices = vec![0; shape.len()];
                    // Set outer dimensions
                    let mut remaining = outer;
                    for d in (0..dim).rev() {
                        indices[d] = remaining % shape[d];
                        remaining /= shape[d];
                    }
                    // Set inner dimensions
                    remaining = inner;
                    for d in (dim + 1..shape.len()).rev() {
                        indices[d] = remaining % shape[d];
                        remaining /= shape[d];
                    }
                    indices[dim] = i;

                    // Convert to flat index
                    let mut flat_idx = 0;
                    let mut stride = 1;
                    for (d, &idx) in indices.iter().enumerate().rev() {
                        flat_idx += idx * stride;
                        stride *= shape[shape.len() - 1 - d];
                    }

                    result_data[flat_idx] = softmax_vector[i];
                }
            }
        }

        let mut result = Tensor::from_vec(result_data, result_shape);

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }
}

impl<T: FloatDtype> fmt::Display for Softmax<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dim == usize::MAX {
            write!(f, "Softmax()")
        } else {
            write!(f, "Softmax(dim={})", self.dim)
        }
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

/// HardTanh activation function
///
/// Formula: `HardTanh(x) = max(min(x, max_val), min_val)`
///
/// HardTanh clips the input to the range [min_val, max_val].
/// Default values are min_val = -1.0, max_val = 1.0.
#[derive(Debug, Clone, Copy)]
pub struct Hardtanh<T: FloatDtype> {
    min_val: T,
    max_val: T,
}

impl<T: FloatDtype> Default for Hardtanh<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> Hardtanh<T> {
    /// Create a new HardTanh activation with default range [-1.0, 1.0]
    pub fn new() -> Self {
        Self {
            min_val: T::from(-1.0).unwrap(),
            max_val: T::from(1.0).unwrap(),
        }
    }

    /// Create a new HardTanh activation with specified range
    pub fn with_range(min_val: T, max_val: T) -> Self {
        Self { min_val, max_val }
    }
}

impl<T: FloatDtype> Module<T> for Hardtanh<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Ok(input.hardtanh(self.min_val, self.max_val))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl<T: FloatDtype> fmt::Display for Hardtanh<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Hardtanh(min_val={}, max_val={})",
            Dtype::to_f64(&self.min_val).unwrap_or(-1.0),
            Dtype::to_f64(&self.max_val).unwrap_or(1.0)
        )
    }
}

/// LogSigmoid activation function
///
/// Formula: `LogSigmoid(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))`
///
/// This is the logarithm of the sigmoid function. It's useful for
/// numerical stability in certain loss functions.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogSigmoid;

impl LogSigmoid {
    /// Create a new LogSigmoid activation
    pub fn new() -> Self {
        Self
    }
}

impl<T: FloatDtype> Module<T> for LogSigmoid {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Ok(input.logsigmoid())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for LogSigmoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LogSigmoid()")
    }
}

/// SELU (Scaled Exponential Linear Unit) activation function
///
/// Formula: `SELU(x) = λ * (x if x > 0 else α * (exp(x) - 1))`
///
/// Where:
/// - λ ≈ 1.0507 (scale parameter)
/// - α ≈ 1.67326 (alpha parameter)
///
/// SELU is a self-normalizing activation function that induces
/// self-normalizing properties in neural networks.
///
/// # References
/// - [Klambauer et al., 2017 - Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
#[derive(Debug, Clone, Copy)]
pub struct SELU<T: FloatDtype> {
    /// Scale parameter λ
    pub lambda: T,
    /// Alpha parameter α
    pub alpha: T,
}

impl<T: FloatDtype> SELU<T> {
    /// Create a new SELU activation with default parameters
    ///
    /// Uses the standard values: λ = 1.0507, α = 1.67326
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::SELU;
    ///
    /// let selu = SELU::<f32>::new();
    /// ```
    pub fn new() -> Self {
        Self {
            lambda: T::from(1.0507).unwrap(),
            alpha: T::from(1.67326).unwrap(),
        }
    }

    /// Create a new SELU activation with custom parameters
    ///
    /// # Arguments
    /// * `lambda` - Scale parameter λ
    /// * `alpha` - Alpha parameter α
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::SELU;
    ///
    /// let selu = SELU::<f32>::with_params(1.0, 1.5);
    /// ```
    pub fn with_params(lambda: T, alpha: T) -> Self {
        Self { lambda, alpha }
    }
}

impl<T: FloatDtype> Default for SELU<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype + std::fmt::Display> fmt::Display for SELU<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SELU(lambda={}, alpha={})", self.lambda, self.alpha)
    }
}

/// CELU (Continuously Differentiable Exponential Linear Unit) activation function
///
/// Formula: `CELU(x) = max(0, x) + min(0, α * (exp(x/α) - 1))`
///
/// CELU is a continuously differentiable version of ELU that maintains
/// the same properties but with smoother derivatives.
///
/// # References
/// - [Barron, 2017 - Continuously Differentiable Exponential Linear Units](https://arxiv.org/abs/1704.07483)
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
        let mut result_data = Vec::with_capacity(input.data().len());

        for &x in input.data() {
            if x >= T::zero() {
                // Positive values: CELU(x) = x
                result_data.push(x);
            } else {
                // Negative values: CELU(x) = α * (exp(x/α) - 1)
                let scaled_x = x / self.alpha;
                let exp_scaled_x = scaled_x.exp();
                result_data.push(self.alpha * (exp_scaled_x - T::one()));
            }
        }

        let mut result = Tensor::from_vec(result_data, input.shape().to_vec());

        // Propagate requires_grad flag
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

/// Hardshrink activation function
///
/// Formula: `Hardshrink(x) = x if |x| > λ else 0`
///
/// Hardshrink sets values within the range [-λ, λ] to zero,
/// while preserving larger values unchanged.
///
/// # References
/// - [PyTorch Hardshrink documentation](https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html)
#[derive(Debug, Clone, Copy)]
pub struct Hardshrink<T: FloatDtype> {
    lambda: T,
}

impl<T: FloatDtype> Default for Hardshrink<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> Hardshrink<T> {
    /// Create a new Hardshrink activation with default lambda = 0.5
    pub fn new() -> Self {
        Self {
            lambda: T::from(0.5).unwrap(),
        }
    }

    /// Create a new Hardshrink activation with specified lambda
    pub fn with_lambda(lambda: T) -> Self {
        Self { lambda }
    }
}

impl<T: FloatDtype> Module<T> for Hardshrink<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let mut result_data = Vec::with_capacity(input.data().len());

        for &x in input.data() {
            if x.abs() > self.lambda {
                // Values outside [-λ, λ]: Hardshrink(x) = x
                result_data.push(x);
            } else {
                // Values within [-λ, λ]: Hardshrink(x) = 0
                result_data.push(T::zero());
            }
        }

        let mut result = Tensor::from_vec(result_data, input.shape().to_vec());

        // Propagate requires_grad flag
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

impl<T: FloatDtype> fmt::Display for Hardshrink<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Hardshrink(lambda={})",
            Dtype::to_f64(&self.lambda).unwrap_or(0.5)
        )
    }
}

/// Tanhshrink activation function
///
/// Formula: `Tanhshrink(x) = x - tanh(x)`
///
/// Tanhshrink subtracts the hyperbolic tangent from the input,
/// creating a function that approaches zero for large inputs.
///
/// # References
/// - [PyTorch Tanhshrink documentation](https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html)
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
        let mut result_data = Vec::with_capacity(input.data().len());

        for &x in input.data() {
            // Tanhshrink(x) = x - tanh(x)
            let tanh_x = x.tanh();
            result_data.push(x - tanh_x);
        }

        let mut result = Tensor::from_vec(result_data, input.shape().to_vec());

        // Propagate requires_grad flag
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

impl fmt::Display for Tanhshrink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tanhshrink()")
    }
}

/// Threshold activation function
///
/// Formula: `Threshold(x) = x if x > threshold else value`
///
/// Threshold replaces values below the threshold with a specified value,
/// while preserving values above the threshold.
///
/// # References
/// - [PyTorch Threshold documentation](https://pytorch.org/docs/stable/generated/torch.nn.Threshold.html)
#[derive(Debug, Clone, Copy)]
pub struct Threshold<T: FloatDtype> {
    threshold: T,
    value: T,
}

impl<T: FloatDtype> Threshold<T> {
    /// Create a new Threshold activation with specified threshold and value
    pub fn new(threshold: T, value: T) -> Self {
        Self { threshold, value }
    }
}

impl<T: FloatDtype> Module<T> for Threshold<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let mut result_data = Vec::with_capacity(input.data().len());

        for &x in input.data() {
            if x > self.threshold {
                // Values above threshold: Threshold(x) = x
                result_data.push(x);
            } else {
                // Values below or equal to threshold: Threshold(x) = value
                result_data.push(self.value);
            }
        }

        let mut result = Tensor::from_vec(result_data, input.shape().to_vec());

        // Propagate requires_grad flag
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

impl<T: FloatDtype> fmt::Display for Threshold<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Threshold(threshold={}, value={})",
            Dtype::to_f64(&self.threshold).unwrap_or(0.0),
            Dtype::to_f64(&self.value).unwrap_or(0.0)
        )
    }
}

impl<T: FloatDtype + Dtype> Module<T> for SELU<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let mut result_data = Vec::with_capacity(input.data().len());

        for &x in input.data() {
            if x > T::zero() {
                // Positive values: SELU(x) = λ * x
                result_data.push(self.lambda * x);
            } else {
                // Negative values: SELU(x) = λ * α * (exp(x) - 1)
                let exp_x_minus_1 = x.exp() - T::one();
                result_data.push(self.lambda * self.alpha * exp_x_minus_1);
            }
        }

        let mut result = Tensor::from_vec(result_data, input.shape().to_vec());

        // Propagate requires_grad flag
        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new()
    }
}

/// PReLU (Parametric Rectified Linear Unit) activation function
///
/// Formula: `PReLU(x) = x if x >= 0 else weight * x`
///
/// PReLU is a learnable activation function that allows different
/// negative slopes for different channels.
///
/// # References
/// - [He et al., 2015 - Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
#[derive(Debug, Clone)]
pub struct PReLU<T: FloatDtype> {
    /// Learnable weights for negative slope (one per channel)
    pub weight: Tensor<T>,
}

impl<T: FloatDtype> PReLU<T> {
    /// Create a new PReLU activation with specified number of channels
    ///
    /// # Arguments
    /// * `num_parameters` - Number of learnable parameters (usually equal to number of channels)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::PReLU;
    ///
    /// let prelu = PReLU::<f32>::new(3); // 3-channel PReLU
    /// ```
    pub fn new(num_parameters: usize) -> Self {
        // Initialize with default negative slope of 0.25
        let weight_data = vec![T::from(0.25).unwrap(); num_parameters];
        let weight = Tensor::from_vec(weight_data, vec![num_parameters]);
        Self { weight }
    }

    /// Create a new PReLU with custom initial weights
    ///
    /// # Arguments
    /// * `weight` - Initial weight tensor
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::PReLU;
    /// use coeus_tensor::Tensor;
    ///
    /// let weight = Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]);
    /// let prelu = PReLU::with_weight(weight);
    /// ```
    pub fn with_weight(weight: Tensor<T>) -> Self {
        Self { weight }
    }
}

impl<T: FloatDtype> Module<T> for PReLU<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let shape = input.shape();
        let mut result_data = Vec::with_capacity(input.numel());

        // For simplicity, handle 1D and 2D cases (extend for full generality)
        if shape.len() == 1 {
            // 1D case: apply same weight to all elements
            let weight_val = self.weight.data()[0];
            for &x in input.data() {
                if x >= T::zero() {
                    result_data.push(x);
                } else {
                    result_data.push(weight_val * x);
                }
            }
        } else if shape.len() == 2 {
            // 2D case: assume channel dimension is last
            let batch_size = shape[0];
            let num_channels = shape[1];

            for b in 0..batch_size {
                for c in 0..num_channels {
                    let idx = b * num_channels + c;
                    let x = input.data()[idx];
                    let weight_val = if c < self.weight.numel() {
                        self.weight.data()[c]
                    } else {
                        self.weight.data()[0] // fallback to first weight
                    };

                    if x >= T::zero() {
                        result_data.push(x);
                    } else {
                        result_data.push(weight_val * x);
                    }
                }
            }
        } else {
            // General case: broadcast weight across channels
            for (i, &x) in input.data().iter().enumerate() {
                let weight_idx = i % self.weight.numel();
                let weight_val = self.weight.data()[weight_idx];

                if x >= T::zero() {
                    result_data.push(x);
                } else {
                    result_data.push(weight_val * x);
                }
            }
        }

        let mut result = Tensor::from_vec(result_data, shape.to_vec());

        // Propagate requires_grad flag
        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.weight]
    }
}

impl<T: FloatDtype> fmt::Display for PReLU<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PReLU(num_parameters={})", self.weight.numel())
    }
}

/// RReLU (Randomized Rectified Linear Unit) activation function
///
/// Formula: `RReLU(x) = x if x >= 0 else a * x` where `a ~ Uniform(lower, upper)`
///
/// RReLU randomly samples the negative slope during training for regularization.
///
/// # References
/// - [Xu et al., 2015 - Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853)
#[derive(Debug, Clone, Copy)]
pub struct RReLU<T: FloatDtype> {
    /// Lower bound for random negative slope
    lower: T,
    /// Upper bound for random negative slope
    upper: T,
    /// Whether in training mode (affects behavior)
    training: bool,
}

impl<T: FloatDtype> RReLU<T> {
    /// Create a new RReLU with default bounds [0.125, 0.333]
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::RReLU;
    ///
    /// let rrelu = RReLU::<f32>::new();
    /// ```
    pub fn new() -> Self {
        Self {
            lower: T::from(0.125).unwrap(),
            upper: T::from(0.333).unwrap(),
            training: true,
        }
    }

    /// Create a new RReLU with custom bounds
    ///
    /// # Arguments
    /// * `lower` - Lower bound for negative slope
    /// * `upper` - Upper bound for negative slope
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::RReLU;
    ///
    /// let rrelu = RReLU::<f32>::with_bounds(0.1, 0.4);
    /// ```
    pub fn with_bounds(lower: T, upper: T) -> Self {
        Self {
            lower,
            upper,
            training: true,
        }
    }

    /// Set training mode
    ///
    /// # Arguments
    /// * `training` - Whether in training mode
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::RReLU;
    ///
    /// let mut rrelu = RReLU::<f32>::new();
    /// rrelu.set_training(false); // Evaluation mode
    /// ```
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl<T: FloatDtype> Default for RReLU<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> Module<T> for RReLU<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let mut result_data = Vec::with_capacity(input.numel());

        if self.training {
            // Training mode: use random negative slope
            use rand::prelude::*;
            let mut rng = rand::thread_rng();

            for &x in input.data() {
                if x >= T::zero() {
                    result_data.push(x);
                } else {
                    // Sample random negative slope in [lower, upper]
                    let random_slope = T::from(rng.gen_range(
                        Dtype::to_f64(&self.lower).unwrap_or(0.125)..
                        Dtype::to_f64(&self.upper).unwrap_or(0.333)
                    )).unwrap_or(self.lower);
                    result_data.push(random_slope * x);
                }
            }
        } else {
            // Evaluation mode: use fixed negative slope (average of bounds)
            let fixed_slope = (self.lower + self.upper) / T::from(2.0).unwrap();

            for &x in input.data() {
                if x >= T::zero() {
                    result_data.push(x);
                } else {
                    result_data.push(fixed_slope * x);
                }
            }
        }

        let mut result = Tensor::from_vec(result_data, input.shape().to_vec());

        // Propagate requires_grad flag
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

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

impl<T: FloatDtype> fmt::Display for RReLU<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RReLU(lower={}, upper={}, training={})",
            Dtype::to_f64(&self.lower).unwrap_or(0.125),
            Dtype::to_f64(&self.upper).unwrap_or(0.333),
            self.training
        )
    }
}

/// Softmin activation function
///
/// Formula: `Softmin(x_i) = exp(-x_i) / sum(exp(-x_j) for all j)`
///
/// Softmin converts a vector of real numbers into a probability distribution
/// where smaller values get higher probabilities.
///
/// # References
/// - [PyTorch Softmin documentation](https://pytorch.org/docs/stable/generated/torch.nn.Softmin.html)
#[derive(Debug, Clone, Copy)]
pub struct Softmin<T: FloatDtype> {
    /// Dimension along which to apply softmin
    dim: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FloatDtype> Default for Softmin<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> Softmin<T> {
    /// Create a new Softmin activation with default dimension (-1, last dimension)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Softmin;
    ///
    /// let softmin = Softmin::<f32>::new();
    /// ```
    pub fn new() -> Self {
        Self {
            dim: usize::MAX,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new Softmin activation along a specific dimension
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to apply softmin
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Softmin;
    ///
    /// let softmin = Softmin::<f32>::new_with_dim(1);
    /// ```
    pub fn new_with_dim(dim: usize) -> Self {
        Self {
            dim,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: FloatDtype + Clone> Module<T> for Softmin<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let dim = if self.dim == usize::MAX {
            input.ndim() - 1
        } else {
            self.dim
        };

        // Softmin(x) = exp(-x) / sum(exp(-x))
        // For now, implement manually since neg() may not be available
        let mut neg_data = Vec::with_capacity(input.data().len());
        for &x in input.data() {
            neg_data.push(T::zero() - x);
        }
        let neg_input = Tensor::from_vec(neg_data, input.shape().to_vec());
        let softmax = Softmax::new_with_dim(dim);
        softmax.forward(&neg_input)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl<T: FloatDtype> fmt::Display for Softmin<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dim == usize::MAX {
            write!(f, "Softmin()")
        } else {
            write!(f, "Softmin(dim={})", self.dim)
        }
    }
}

/// Softmax2d activation function
///
/// Formula: `Softmax2d(x_i) = exp(x_i) / sum(exp(x_j) for all j in the same channel)`
///
/// Softmax2d applies softmax along the channel dimension for 4D tensors.
/// This is commonly used in segmentation tasks.
///
/// # References
/// - [PyTorch Softmax2d documentation](https://pytorch.org/docs/stable/generated/torch.nn.Softmax2d.html)
#[derive(Debug, Clone, Copy, Default)]
pub struct Softmax2d;

impl Softmax2d {
    /// Create a new Softmax2d activation
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Softmax2d;
    ///
    /// let softmax2d = Softmax2d::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl<T: FloatDtype> Module<T> for Softmax2d {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 4 {
            return Err(crate::NNError::InvalidInput {
                message: format!("Softmax2d requires 4D input, got {}D", input.ndim()),
            });
        }

        let shape = input.shape();
        let n = shape[0]; // batch size
        let c = shape[1]; // channels
        let h = shape[2]; // height
        let w = shape[3]; // width

        let mut result_data = vec![T::zero(); input.numel()];

        // Apply softmax along channel dimension for each spatial location
        for batch in 0..n {
            for height in 0..h {
                for width in 0..w {
                    // Extract channel values for this spatial location
                    let mut channel_values = vec![T::zero(); c];
                    #[allow(clippy::needless_range_loop)]
                    for channel in 0..c {
                        let idx = ((batch * c + channel) * h + height) * w + width;
                        channel_values[channel] = input.data()[idx];
                    }

                    // Compute softmax for these channel values
                    let max_val = channel_values.iter().fold(T::from(f64::NEG_INFINITY).unwrap(), |a, &b| a.max(b));
                    let exp_values: Vec<T> = channel_values.iter().map(|&x| (x - max_val).exp()).collect();
                    let sum_exp: T = exp_values.iter().fold(T::zero(), |a, &b| a + b);

                    // Store softmax results
                    #[allow(clippy::needless_range_loop)]
                    for channel in 0..c {
                        let idx = ((batch * c + channel) * h + height) * w + width;
                        result_data[idx] = exp_values[channel] / sum_exp;
                    }
                }
            }
        }

        let mut result = Tensor::from_vec(result_data, shape.to_vec());

        // Propagate requires_grad flag
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

impl fmt::Display for Softmax2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Softmax2d()")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_elu_forward() {
        let elu = ELU::<f32>::new();

        // Test basic ELU behavior
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = elu.forward(&input).unwrap();

        // ELU(x) = x if x > 0 else α * (exp(x) - 1), α = 1.0
        assert_relative_eq!(
            output.data()[0],
            1.0 * ((-2.0f32).exp() - 1.0),
            epsilon = 1e-6
        ); // x = -2
        assert_relative_eq!(
            output.data()[1],
            1.0 * ((-1.0f32).exp() - 1.0),
            epsilon = 1e-6
        ); // x = -1
        assert_relative_eq!(output.data()[2], 0.0, epsilon = 1e-6); // x = 0
        assert_relative_eq!(output.data()[3], 1.0, epsilon = 1e-6); // x = 1
        assert_relative_eq!(output.data()[4], 2.0, epsilon = 1e-6); // x = 2
    }

    #[test]
    fn test_elu_with_alpha() {
        let elu = ELU::with_alpha(2.0);

        let input = Tensor::from_vec(vec![-1.0, 1.0], vec![2]);
        let output = elu.forward(&input).unwrap();

        // ELU(x) = x if x > 0 else α * (exp(x) - 1), α = 2.0
        assert_relative_eq!(
            output.data()[0],
            2.0 * ((-1.0f32).exp() - 1.0),
            epsilon = 1e-6
        ); // x = -1
        assert_relative_eq!(output.data()[1], 1.0, epsilon = 1e-6); // x = 1
    }

    #[test]
    fn test_gelu_forward() {
        let gelu = GELU::new();

        let input = Tensor::from_vec(vec![0.0], vec![1]);
        let output = gelu.forward(&input).unwrap();

        // GELU(0) should be approximately 0
        assert_relative_eq!(output.data()[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hardtanh_forward() {
        let hardtanh = Hardtanh::<f32>::new();

        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = hardtanh.forward(&input).unwrap();

        // HardTanh should clip to [-1, 1]
        assert_relative_eq!(output.data()[0], -1.0, epsilon = 1e-6); // x = -2 -> clipped to -1
        assert_relative_eq!(output.data()[1], -1.0, epsilon = 1e-6); // x = -1 -> within range
        assert_relative_eq!(output.data()[2], 0.0, epsilon = 1e-6); // x = 0 -> within range
        assert_relative_eq!(output.data()[3], 1.0, epsilon = 1e-6); // x = 1 -> within range
        assert_relative_eq!(output.data()[4], 1.0, epsilon = 1e-6); // x = 2 -> clipped to 1
    }

    #[test]
    fn test_hardtanh_with_range() {
        let hardtanh = Hardtanh::with_range(-2.0, 2.0);

        let input = Tensor::from_vec(vec![-3.0, 0.0, 3.0], vec![3]);
        let output = hardtanh.forward(&input).unwrap();

        assert_relative_eq!(output.data()[0], -2.0, epsilon = 1e-6); // x = -3 -> clipped to -2
        assert_relative_eq!(output.data()[1], 0.0, epsilon = 1e-6); // x = 0 -> within range
        assert_relative_eq!(output.data()[2], 2.0, epsilon = 1e-6); // x = 3 -> clipped to 2
    }

    #[test]
    fn test_logsigmoid_forward() {
        let logsigmoid = LogSigmoid::new();

        let input = Tensor::from_vec(vec![0.0], vec![1]);
        let output = logsigmoid.forward(&input).unwrap();

        // LogSigmoid(0) = log(sigmoid(0)) = log(0.5) ≈ -0.693
        assert_relative_eq!(output.data()[0], -std::f64::consts::LN_2, epsilon = 1e-6);
    }

    #[test]
    fn test_activation_module_trait() {
        // Test that all activation functions implement Module trait correctly
        let mut elu = ELU::<f32>::new();
        let mut gelu = GELU::new();
        let mut hardtanh = Hardtanh::<f32>::new();
        let mut logsigmoid = LogSigmoid::new();

        // All should have no parameters
        assert_eq!(elu.parameters().len(), 0);
        assert_eq!(<GELU as Module<f32>>::parameters(&gelu).len(), 0);
        assert_eq!(hardtanh.parameters().len(), 0);
        assert_eq!(
            <LogSigmoid as Module<f32>>::parameters(&logsigmoid).len(),
            0
        );

        assert_eq!(elu.parameters_mut().len(), 0);
        assert_eq!(<GELU as Module<f32>>::parameters_mut(&mut gelu).len(), 0);
        assert_eq!(hardtanh.parameters_mut().len(), 0);
        assert_eq!(
            <LogSigmoid as Module<f32>>::parameters_mut(&mut logsigmoid).len(),
            0
        );
    }

    #[test]
    fn test_relu_forward() {
        let relu = ReLU::new();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let output = relu.forward(&input).expect("ReLU forward should succeed");

        let expected = [0.0, 0.0, 1.0, 2.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            assert_relative_eq!(output.data()[i], expected_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_forward() {
        let sigmoid = Sigmoid::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]);
        let output = sigmoid
            .forward(&input)
            .expect("Sigmoid forward should succeed");

        // σ(0) = 0.5
        assert_relative_eq!(output.data()[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_forward() {
        let tanh = Tanh::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]);
        let output = tanh.forward(&input).expect("Tanh forward should succeed");

        // tanh(0) = 0
        assert_relative_eq!(output.data()[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_1d() {
        let softmax = Softmax::new();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let output = softmax
            .forward(&input)
            .expect("Softmax forward should succeed");

        // Check that output sums to 1
        let sum: f32 = output.data().iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Check that all values are positive
        for &val in output.data() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_softmax_2d() {
        let softmax = Softmax::new();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let output = softmax
            .forward(&input)
            .expect("Softmax forward should succeed");

        // Check that each row sums to 1
        for row in 0..2 {
            let start = row * 3;
            let end = (row + 1) * 3;
            let sum: f32 = output.data()[start..end].iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_leaky_relu() {
        let leaky_relu = LeakyReLU::new_with_slope(0.1);
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = leaky_relu
            .forward(&input)
            .expect("LeakyReLU forward should succeed");

        let expected = [-0.2, -0.1, 0.0, 1.0, 2.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            assert_relative_eq!(output.data()[i], expected_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_activation_gradient_flow() {
        let relu = ReLU::new();
        let input = Tensor::from_vec_with_grad(vec![-1.0, 0.5, 2.0], vec![3]);
        let output = relu.forward(&input).expect("ReLU forward should succeed");

        // Compute some loss
        let loss = output.sum();

        // Call backward and then manually check the autograd context
        let _ = loss.backward();

        // For now, check that the loss tensor has a gradient (it should)
        assert!(loss.grad().is_some());

        // The input gradient check is more complex due to current autograd limitations
        // We'll verify this works when we complete the autograd system
        // For now, just ensure the computation doesn't panic
        println!("Activation gradient flow test completed without panic");
    }

    #[test]
    fn test_selu_forward() {
        let selu = SELU::<f32>::new();
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = selu.forward(&input).expect("SELU forward should succeed");

        // Test positive values: SELU(x) = λ * x
        let lambda = 1.0507;
        assert_relative_eq!(output.data()[3], lambda * 1.0, epsilon = 1e-4);
        assert_relative_eq!(output.data()[4], lambda * 2.0, epsilon = 1e-4);

        // Test negative values: SELU(x) = λ * α * (exp(x) - 1)
        let alpha = 1.67326;
        let expected_neg2 = lambda * alpha * ((-2.0f32).exp() - 1.0);
        let expected_neg1 = lambda * alpha * ((-1.0f32).exp() - 1.0);

        assert_relative_eq!(output.data()[0], expected_neg2, epsilon = 1e-4);
        assert_relative_eq!(output.data()[1], expected_neg1, epsilon = 1e-4);

        // Test zero: SELU(0) = λ * 0 = 0
        assert_relative_eq!(output.data()[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_selu_with_custom_params() {
        let lambda = 2.0;
        let alpha = 3.0;
        let selu = SELU::<f32>::with_params(lambda, alpha);
        let input = Tensor::from_vec(vec![-1.0, 1.0], vec![2]);
        let output = selu.forward(&input).expect("SELU forward should succeed");

        // Test positive value: SELU(x) = λ * x
        assert_relative_eq!(output.data()[1], lambda * 1.0, epsilon = 1e-6);

        // Test negative value: SELU(x) = λ * α * (exp(x) - 1)
        let expected = lambda * alpha * ((-1.0f32).exp() - 1.0);
        assert_relative_eq!(output.data()[0], expected, epsilon = 1e-6);
    }

    #[test]
    fn test_celu_forward() {
        let celu = CELU::<f32>::new();
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = celu.forward(&input).expect("CELU forward should succeed");

        // CELU(x) = x if x > 0 else α * (exp(x/α) - 1), α = 1.0
        let alpha = 1.0;
        assert_relative_eq!(
            output.data()[0],
            alpha * ((-2.0f32 / alpha).exp() - 1.0),
            epsilon = 1e-6
        ); // x = -2
        assert_relative_eq!(
            output.data()[1],
            alpha * ((-1.0f32 / alpha).exp() - 1.0),
            epsilon = 1e-6
        ); // x = -1
        assert_relative_eq!(output.data()[2], 0.0, epsilon = 1e-6); // x = 0
        assert_relative_eq!(output.data()[3], 1.0, epsilon = 1e-6); // x = 1
        assert_relative_eq!(output.data()[4], 2.0, epsilon = 1e-6); // x = 2
    }

    #[test]
    fn test_hardshrink_forward() {
        let hardshrink = Hardshrink::<f32>::new();
        let input = Tensor::from_vec(vec![-1.0, -0.3, 0.0, 0.3, 1.0], vec![5]);
        let output = hardshrink
            .forward(&input)
            .expect("Hardshrink forward should succeed");

        // Hardshrink(x) = x if |x| > λ else 0, λ = 0.5
        assert_relative_eq!(output.data()[0], -1.0, epsilon = 1e-6); // |x| = 1.0 > 0.5, preserved
        assert_relative_eq!(output.data()[1], 0.0, epsilon = 1e-6); // |x| = 0.3 < 0.5, set to 0
        assert_relative_eq!(output.data()[2], 0.0, epsilon = 1e-6); // x = 0
        assert_relative_eq!(output.data()[3], 0.0, epsilon = 1e-6); // |x| = 0.3 < 0.5, set to 0
        assert_relative_eq!(output.data()[4], 1.0, epsilon = 1e-6); // |x| = 1.0 > 0.5, preserved
    }

    #[test]
    fn test_tanhshrink_forward() {
        let tanhshrink = Tanhshrink::new();
        let input = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
        let output = tanhshrink
            .forward(&input)
            .expect("Tanhshrink forward should succeed");

        // Tanhshrink(x) = x - tanh(x)
        assert_relative_eq!(output.data()[0], 0.0 - 0.0f32.tanh(), epsilon = 1e-6); // x = 0
        assert_relative_eq!(output.data()[1], 1.0 - 1.0f32.tanh(), epsilon = 1e-6);
        // x = 1
    }

    #[test]
    fn test_threshold_forward() {
        let threshold = Threshold::<f32>::new(0.5, -1.0);
        let input = Tensor::from_vec(vec![0.0, 0.3, 0.5, 0.7, 1.0], vec![5]);
        let output = threshold
            .forward(&input)
            .expect("Threshold forward should succeed");

        // Threshold(x) = x if x > threshold else value
        assert_relative_eq!(output.data()[0], -1.0, epsilon = 1e-6); // x = 0.0 <= 0.5, set to -1
        assert_relative_eq!(output.data()[1], -1.0, epsilon = 1e-6); // x = 0.3 <= 0.5, set to -1
        assert_relative_eq!(output.data()[2], -1.0, epsilon = 1e-6); // x = 0.5 <= 0.5, set to -1 (note: > threshold, not >=)
        assert_relative_eq!(output.data()[3], 0.7, epsilon = 1e-6); // x = 0.7 > 0.5, preserved
        assert_relative_eq!(output.data()[4], 1.0, epsilon = 1e-6); // x = 1.0 > 0.5, preserved
    }

    #[test]
    fn test_prelu_forward() {
        let prelu = PReLU::<f32>::new(3);
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = prelu.forward(&input).expect("PReLU forward should succeed");

        // PReLU(x) = x if x >= 0 else weight * x
        // With default weight = 0.25 for all channels
        assert_relative_eq!(output.data()[0], -2.0 * 0.25, epsilon = 1e-6); // x = -2
        assert_relative_eq!(output.data()[1], -0.25, epsilon = 1e-6); // x = -1
        assert_relative_eq!(output.data()[2], 0.0, epsilon = 1e-6); // x = 0
        assert_relative_eq!(output.data()[3], 1.0, epsilon = 1e-6); // x = 1
        assert_relative_eq!(output.data()[4], 2.0, epsilon = 1e-6); // x = 2
    }

    #[test]
    fn test_softmin_forward() {
        let softmin = Softmin::<f32>::new();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let output = softmin.forward(&input).expect("Softmin forward should succeed");

        // Check that output sums to 1
        let sum: f32 = output.data().iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Check that all values are positive
        for &val in output.data() {
            assert!(val > 0.0);
        }
    }
}
