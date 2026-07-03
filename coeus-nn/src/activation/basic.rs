use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, Scalar};

/// Functional ReLU activation.
#[inline]
pub fn relu<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::relu(input)
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

/// Functional Sigmoid activation.
#[inline]
pub fn sigmoid<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::sigmoid(input)
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

/// Functional Tanh activation.
#[inline]
pub fn tanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::tanh(input)
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

/// Functional GELU activation.
#[inline]
pub fn gelu<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::gelu(input)
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

/// Public descriptor for the functional Hardsigmoid operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HardsigmoidOp;

impl HardsigmoidOp {
    /// Operation name used by the autograd node.
    pub const OP_NAME: &'static str = "hardsigmoid";
}

/// Functional Hardsigmoid activation.
#[inline]
pub fn hardsigmoid<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::hardsigmoid(input)
}

/// Hardsigmoid activation module.
#[derive(Clone, Debug, Default)]
pub struct Hardsigmoid;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Hardsigmoid {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        hardsigmoid(input)
    }
}

/// Public descriptor for the functional Hardswish operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HardswishOp;

impl HardswishOp {
    /// Operation name used by the autograd node.
    pub const OP_NAME: &'static str = "hardswish";
}

/// Functional Hardswish activation.
#[inline]
pub fn hardswish<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::hardswish(input)
}

/// Hardswish activation module.
#[derive(Clone, Debug, Default)]
pub struct Hardswish;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Hardswish {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        hardswish(input)
    }
}

/// Public descriptor for the functional Softsign operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftsignOp;

impl SoftsignOp {
    /// Operation name used by the autograd node.
    pub const OP_NAME: &'static str = "softsign";
}

/// Functional Softsign activation.
#[inline]
pub fn softsign<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::softsign(input)
}

/// Softsign activation module.
#[derive(Clone, Debug, Default)]
pub struct Softsign;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Softsign {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        softsign(input)
    }
}

/// Functional Softplus activation.
#[inline]
pub fn softplus<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
) -> Var<T, B> {
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

/// Functional LogSigmoid activation: `log(sigmoid(x)) = -softplus(-x)`.
///
/// Uses the numerically stable `-softplus(-x)` identity (avoids `log` of a tiny
/// sigmoid); matches `torch.nn.functional.logsigmoid`.
#[inline]
pub fn log_sigmoid<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::neg(&coeus_autograd::softplus(&coeus_autograd::neg(input)))
}

/// LogSigmoid activation module.
#[derive(Clone, Debug, Default)]
pub struct LogSigmoid;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for LogSigmoid {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        log_sigmoid(input)
    }
}

/// Functional Tanhshrink activation: `x - tanh(x)`.
///
/// Matches `torch.nn.functional.tanhshrink`.
#[inline]
pub fn tanhshrink<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::sub(input, &coeus_autograd::tanh(input))
}

/// Tanhshrink activation module.
#[derive(Clone, Debug, Default)]
pub struct Tanhshrink;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Tanhshrink {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        tanhshrink(input)
    }
}
