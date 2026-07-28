//! Basic activation functions and zero-parameter activation modules.
//!
//! This module exposes both functional helpers, such as [`relu`] and
//! [`sigmoid`], and corresponding [`crate::module::Module`] wrappers for
//! building models.

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, Scalar};

/// Functional ReLU activation.
#[inline]
pub fn relu<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::relu(input)
}

/// ReLU activation module.
#[derive(Clone, Debug, Default)]
pub struct ReLU;

/// Implements the [`crate::module::Module`] interface for [`ReLU`].
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for ReLU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        relu(input)
    }
}

/// Functional Sigmoid activation.
#[inline]
pub fn sigmoid<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::sigmoid(input)
}

/// Sigmoid activation module.
#[derive(Clone, Debug, Default)]
pub struct Sigmoid;

/// Implements the [`crate::module::Module`] interface for [`Sigmoid`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Sigmoid {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        sigmoid(input)
    }
}

/// Functional Tanh activation.
#[inline]
pub fn tanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::tanh(input)
}

/// Tanh activation module.
#[derive(Clone, Debug, Default)]
pub struct Tanh;

/// Implements the [`crate::module::Module`] interface for [`Tanh`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Tanh {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        tanh(input)
    }
}

/// Functional GELU activation.
#[inline]
pub fn gelu<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::gelu(input)
}

/// GeLU activation module.
#[derive(Clone, Debug, Default)]
pub struct GeLU;

/// Implements the [`crate::module::Module`] interface for [`GeLU`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GeLU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        gelu(input)
    }
}

/// Functional SiLU activation.
#[inline]
pub fn silu<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::silu(input)
}

/// SiLU activation module.
#[derive(Clone, Debug, Default)]
pub struct SiLU;

/// Implements the [`crate::module::Module`] interface for [`SiLU`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for SiLU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        silu(input)
    }
}

/// Functional Mish activation.
#[inline]
pub fn mish<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::mish(input)
}

/// Mish activation module.
#[derive(Clone, Debug, Default)]
pub struct Mish;

/// Implements the [`crate::module::Module`] interface for [`Mish`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Mish {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
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
) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::hardsigmoid(input)
}

/// Hardsigmoid activation module.
#[derive(Clone, Debug, Default)]
pub struct Hardsigmoid;

/// Implements the [`crate::module::Module`] interface for [`Hardsigmoid`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Hardsigmoid {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
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
pub fn hardswish<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::hardswish(input)
}

/// Hardswish activation module.
#[derive(Clone, Debug, Default)]
pub struct Hardswish;

/// Implements the [`crate::module::Module`] interface for [`Hardswish`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Hardswish {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
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
pub fn softsign<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::softsign(input)
}

/// Softsign activation module.
#[derive(Clone, Debug, Default)]
pub struct Softsign;

/// Implements the [`crate::module::Module`] interface for [`Softsign`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Softsign {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        softsign(input)
    }
}

/// Functional Softplus activation.
#[inline]
pub fn softplus<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::softplus(input)
}

/// Softplus activation module.
#[derive(Clone, Debug, Default)]
pub struct Softplus;

/// Implements the [`crate::module::Module`] interface for [`Softplus`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Softplus {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
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
) -> Result<Var<T, B>, B::Error> {
    let negated = coeus_autograd::neg(input)?;
    let softplus = coeus_autograd::softplus(&negated)?;
    coeus_autograd::neg(&softplus)
}

/// LogSigmoid activation module.
#[derive(Clone, Debug, Default)]
pub struct LogSigmoid;

/// Implements the [`crate::module::Module`] interface for [`LogSigmoid`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for LogSigmoid {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        log_sigmoid(input)
    }
}

/// Functional Tanhshrink activation: `x - tanh(x)`.
///
/// Matches `torch.nn.functional.tanhshrink`.
#[inline]
pub fn tanhshrink<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
    let tangent = coeus_autograd::tanh(input)?;
    coeus_autograd::sub(input, &tangent)
}

/// Tanhshrink activation module.
#[derive(Clone, Debug, Default)]
pub struct Tanhshrink;

/// Implements the [`crate::module::Module`] interface for [`Tanhshrink`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Tanhshrink {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        tanhshrink(input)
    }
}
