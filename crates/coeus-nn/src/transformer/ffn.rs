// ── Feed-Forward Network sub-layer ──

use crate::dropout::Dropout;
use crate::linear::Linear;
use crate::module::{prefixed_parameters, Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use std::marker::PhantomData;

/// Functional (stateless) transformer feed-forward block.
///
/// Computes: `Linear1(d_model→d_ff) → GELU → Dropout(p) → Linear2(d_ff→d_model)`.
pub fn feed_forward<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    w1: &Var<T, B>,
    b1: Option<&Var<T, B>>,
    w2: &Var<T, B>,
    b2: Option<&Var<T, B>>,
    dropout_p: f64,
) -> Result<Var<T, B>, ModuleError<B::Error>> {
    feed_forward_with_training(input, w1, b1, w2, b2, dropout_p, dropout_p > 0.0)
}

pub(super) fn feed_forward_with_training<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    w1: &Var<T, B>,
    b1: Option<&Var<T, B>>,
    w2: &Var<T, B>,
    b2: Option<&Var<T, B>>,
    dropout_p: f64,
    is_training: bool,
) -> Result<Var<T, B>, ModuleError<B::Error>> {
    let shape = input.tensor.shape();
    if shape.len() < 2 {
        return Err(ModuleError::InvalidRank {
            module: "FeedForward",
            expected: "at least 2",
            actual: shape.len(),
        });
    }
    let d_model = *shape
        .last()
        .expect("invariant: feed-forward rank was validated as at least two");
    super::validation::feed_forward("FeedForward", d_model, w1, b1, w2, b2)?;

    let x = linear_from_parts(input, w1, b1);
    let x = coeus_autograd::gelu(&x);
    let mut dropout = Dropout::new(dropout_p);
    dropout.set_training(is_training);
    let x = dropout.forward(&x)?;
    Ok(linear_from_parts(&x, w2, b2))
}

fn linear_from_parts<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: Option<&Var<T, B>>,
) -> Var<T, B> {
    let input_shape = input.tensor.shape();
    let in_features = weight.tensor.shape()[1];
    let out_features = weight.tensor.shape()[0];
    let rows = input_shape[..input_shape.len() - 1]
        .iter()
        .copied()
        .product::<usize>();
    let flattened = coeus_autograd::reshape(input, [rows, in_features]);
    let weight_transposed = coeus_autograd::transpose_2d(weight);
    let projected = coeus_autograd::matmul(&flattened, &weight_transposed);
    let projected = match bias {
        Some(bias) => coeus_autograd::add(&projected, bias),
        None => projected,
    };

    let mut output_shape = input_shape.to_vec();
    *output_shape
        .last_mut()
        .expect("invariant: feed-forward rank was validated as at least two") = out_features;
    coeus_autograd::reshape(&projected, output_shape)
}

/// Two-layer feed-forward sub-layer.
///
/// Computes: `Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)`.
///
/// Accepts inputs of any rank ≥ 2 (`[..., d_model]`), including the standard
/// transformer shape `[batch, seq, d_model]`, via the batched matmul support
/// in `coeus_autograd::matmul`.
#[derive(Clone)]
pub struct FeedForward<T: coeus_core::Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend>
{
    /// First linear projection: `d_model → d_ff`.
    pub linear1: Linear<T, B>,
    /// Second linear projection: `d_ff → d_model`.
    pub linear2: Linear<T, B>,
    /// Dropout between the two projections.
    pub dropout: Dropout,
    _marker: PhantomData<(T, B)>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> FeedForward<T, B> {
    /// Create a FeedForward sub-layer.
    ///
    /// # Arguments
    /// - `d_model`: input and output feature dimension
    /// - `d_ff`:    hidden (inner) feature dimension
    /// - `dropout_p`: dropout probability
    ///
    /// # Errors
    ///
    /// Returns [`crate::init::InitializationError`] when a size is zero or the
    /// backend's draw fails.
    pub fn new(
        d_model: usize,
        d_ff: usize,
        dropout_p: f64,
    ) -> Result<Self, crate::init::InitializationError<B::Error>>
    where
        T: coeus_leto::RandomScalar,
        B: coeus_ops::RandomInitOps<T>,
    {
        Ok(Self {
            linear1: Linear::new(d_model, d_ff, true)?,
            linear2: Linear::new(d_ff, d_model, true)?,
            dropout: Dropout::new(dropout_p),
            _marker: PhantomData,
        })
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for FeedForward<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = self.linear1.parameters();
        p.extend(self.linear2.parameters());
        p
    }

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<T, B>> {
        let mut parameters = prefixed_parameters("input", &self.linear1);
        parameters.extend(prefixed_parameters("output", &self.linear2));
        parameters
    }

    /// Forward pass: `Linear1 → GELU → Dropout → Linear2`.
    ///
    /// Works for any input rank (`[batch, seq, d_model]`, `[batch, d_model]`, etc.)
    /// because `Linear::forward` dispatches to the batched matmul kernel.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        feed_forward(
            input,
            &self.linear1.weight,
            self.linear1.bias.as_ref(),
            &self.linear2.weight,
            self.linear2.bias.as_ref(),
            self.dropout.p,
        )
    }
}
