// ── Feed-Forward Network sub-layer ──

use std::marker::PhantomData;
use coeus_core::{Float, MoiraiBackend};
use coeus_autograd::Var;
use crate::module::Module;
use crate::linear::Linear;
use crate::dropout::Dropout;

/// Two-layer feed-forward sub-layer.
///
/// Computes: `Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)`.
///
/// Accepts inputs of any rank ≥ 2 (`[..., d_model]`), including the standard
/// transformer shape `[batch, seq, d_model]`, via the batched matmul support
/// in `coeus_autograd::matmul`.
#[derive(Clone)]
pub struct FeedForward<
    T: coeus_core::Scalar,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
> {
    pub linear1: Linear<T, B>,
    pub linear2: Linear<T, B>,
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
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> Self {
        Self {
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
            dropout: Dropout::new(dropout_p),
            _marker: PhantomData,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for FeedForward<T, B>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = self.linear1.parameters();
        p.extend(self.linear2.parameters());
        p
    }

    /// Forward pass: `Linear1 → GELU → Dropout → Linear2`.
    ///
    /// Works for any input rank (`[batch, seq, d_model]`, `[batch, d_model]`, etc.)
    /// because `Linear::forward` dispatches to the batched matmul kernel.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let x = self.linear1.forward(input);
        let x = coeus_autograd::gelu(&x);
        let x = self.dropout.forward(&x);
        self.linear2.forward(&x)
    }
}

