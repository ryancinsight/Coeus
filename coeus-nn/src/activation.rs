// ── Activation functions (NN wrappers and modules) ──

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, Scalar};

/// Functional ReLU activation.
#[inline]
pub fn relu<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::relu(input)
}

/// Functional Sigmoid activation.
#[inline]
pub fn sigmoid<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::sigmoid(input)
}

/// Functional Tanh activation.
#[inline]
pub fn tanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::tanh(input)
}

/// Functional GELU activation.
#[inline]
pub fn gelu<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::gelu(input)
}

/// ReLU activation module.
#[derive(Clone, Debug, Default)]
pub struct ReLU;

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for ReLU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        relu(input)
    }
}

/// Sigmoid activation module.
#[derive(Clone, Debug, Default)]
pub struct Sigmoid;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Sigmoid {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        sigmoid(input)
    }
}

/// Tanh activation module.
#[derive(Clone, Debug, Default)]
pub struct Tanh;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Tanh {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        tanh(input)
    }
}

/// GeLU activation module.
#[derive(Clone, Debug, Default)]
pub struct GeLU;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GeLU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        gelu(input)
    }
}

/// Functional SiLU activation.
#[inline]
pub fn silu<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::silu(input)
}

/// SiLU activation module.
#[derive(Clone, Debug, Default)]
pub struct SiLU;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for SiLU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        silu(input)
    }
}

/// Functional Mish activation.
#[inline]
pub fn mish<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::mish(input)
}

/// Mish activation module.
#[derive(Clone, Debug, Default)]
pub struct Mish;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Mish {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        mish(input)
    }
}

// ── Phase 7 New Activations ──

/// Functional ELU activation.
#[inline]
pub fn elu<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::elu(input)
}

/// ELU activation module (alpha = 1.0).
#[derive(Clone, Debug, Default)]
pub struct ELU;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for ELU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }
    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        elu(input)
    }
}

/// Functional Softplus activation.
#[inline]
pub fn softplus<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::softplus(input)
}

/// Softplus activation module.
#[derive(Clone, Debug, Default)]
pub struct Softplus;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Softplus {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }
    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        softplus(input)
    }
}

/// Functional GELU tanh approximation.
#[inline]
pub fn gelu_tanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::gelu_tanh(input)
}

/// GELU tanh approximation module.
#[derive(Clone, Debug, Default)]
pub struct GeLUTanh;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GeLUTanh {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }
    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        gelu_tanh(input)
    }
}

/// Functional LeakyReLU activation.
#[inline]
pub fn leaky_relu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    negative_slope: f64,
) -> Var<T, B> {
    coeus_autograd::leaky_relu(input, negative_slope)
}

/// LeakyReLU activation module.
#[derive(Clone, Debug)]
pub struct LeakyReLU {
    /// Slope for negative inputs.
    pub negative_slope: f64,
}

impl LeakyReLU {
    /// Create a LeakyReLU module.
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for LeakyReLU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }
    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        leaky_relu(input, self.negative_slope)
    }
}
